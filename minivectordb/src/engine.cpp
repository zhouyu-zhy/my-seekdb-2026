#include "mvdb.h"
#include "distance.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <limits>
#include <queue>
#include <set>
#include <unordered_set>
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif

namespace mvdb {

namespace {

std::string lower(const std::string& s) {
    std::string out = s;
    for (char& c : out) {
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    }
    return out;
}

bool ieq(const std::string& a, const std::string& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        char x = a[i], y = b[i];
        if (x >= 'A' && x <= 'Z') x = static_cast<char>(x - 'A' + 'a');
        if (y >= 'A' && y <= 'Z') y = static_cast<char>(y - 'A' + 'a');
        if (x != y) return false;
    }
    return true;
}

size_t utf8Length(const std::string& s) {
    size_t n = 0;
    for (unsigned char c : s) {
        if ((c & 0xC0) != 0x80) ++n;
    }
    return n;
}

Value valueForColumn(const Column& c, const Literal& lit) {
    Value v;
    if (lit.kind == Literal::NULLV) fail("unsupported_sql", "NULL values are not supported");
    if (c.type == ColType::BIGINT) {
        if (lit.kind != Literal::INT) {
            if (lit.kind == Literal::DOUBLE) fail("invalid_number", "BIGINT requires an integer");
            fail("type_mismatch", "BIGINT requires an integer");
        }
        v.i = lit.i;
    } else if (c.type == ColType::VARCHAR) {
        if (lit.kind != Literal::STR) fail("type_mismatch", "VARCHAR requires a string");
        if (utf8Length(lit.s) > c.length) fail("value_too_long", "string exceeds VARCHAR length");
        v.s = lit.s;
    } else {
        if (lit.kind == Literal::STR) {
            v.v = parseVectorTextFloat(lit.s, c.length);
            if (v.v.size() != c.length) fail("dimension_mismatch", "vector dimension does not match column");
            return v;
        }
        std::vector<double> nums;
        if (lit.kind == Literal::VECLIT) {
            nums = lit.v;
        } else {
            fail("type_mismatch", "VECTOR requires an array literal");
        }
        if (nums.size() != c.length) fail("dimension_mismatch", "vector dimension does not match column");
        v.v.reserve(nums.size());
        for (double d : nums) {
            if (!std::isfinite(d) || std::fabs(d) > static_cast<double>(FLT_MAX)) {
                fail("invalid_vector", "vector component is not a finite float32");
            }
            float f = static_cast<float>(d);
            if (!std::isfinite(f)) fail("invalid_vector", "vector component overflows float32");
            v.v.push_back(f);
        }
    }
    return v;
}

std::vector<float> makeQueryVector(const Column& c, const std::vector<double>& nums) {
    if (nums.size() != c.length) fail("dimension_mismatch", "query vector dimension does not match column");
    std::vector<float> out;
    out.reserve(nums.size());
    for (double d : nums) {
        if (!std::isfinite(d) || std::fabs(d) > static_cast<double>(FLT_MAX)) {
            fail("invalid_vector", "vector component is not a finite float32");
        }
        float f = static_cast<float>(d);
        if (!std::isfinite(f)) fail("invalid_vector", "vector component overflows float32");
        out.push_back(f);
    }
    return out;
}

double l2Distance(const std::vector<float>& a, const std::vector<float>& b) {
    return l2DistancePtr(a.data(), b.data(), a.size());
}

// The float pass only rejects candidates that are provably above the current
// threshold. Every remaining candidate follows the original double path,
// including overflow, near ties, and dimensions beyond the error budget.
bool outsideTopK(const float* a, const float* b, size_t n, double squaredCutoff) {
#if defined(__AVX2__) && defined(__FMA__)
    if (n < 32 || n > 4096) return false;
    __m256 acc0 = _mm256_setzero_ps(), acc1 = acc0, acc2 = acc0, acc3 = acc0;
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        const __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        const __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8));
        const __m256 d2 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16));
        const __m256 d3 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24));
        acc0 = _mm256_fmadd_ps(d0, d0, acc0);
        acc1 = _mm256_fmadd_ps(d1, d1, acc1);
        acc2 = _mm256_fmadd_ps(d2, d2, acc2);
        acc3 = _mm256_fmadd_ps(d3, d3, acc3);
    }
    const __m256 acc = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    const __m128 half = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    const __m128 pairs = _mm_add_ps(half, _mm_movehl_ps(half, half));
    float sum = _mm_cvtss_f32(_mm_add_ss(pairs, _mm_shuffle_ps(pairs, pairs, 1)));
    for (; i < n; ++i) {
        const float d = a[i] - b[i];
        sum = d * d + sum;
    }
    // At most n rounded additions plus three operations per term. Use a
    // wider envelope than that bound and allow for underflow in each term.
    const double slack = (static_cast<double>(n) + 16) * 4 * FLT_EPSILON;
    return std::isfinite(sum) &&
           static_cast<double>(sum) > squaredCutoff * (1 + slack) + static_cast<double>(n) * 1e-36;
#else
    (void)a; (void)b; (void)n; (void)squaredCutoff;
    return false;
#endif
}

#if defined(__AVX2__) && defined(__FMA__)
// Dataset coordinates use signed 7-bit integers, queries use unsigned
// 8-bit integers at half the scale. This permits a non-saturating AVX2
// byte dot product. Keep the actual maximum residual norm of the rows
// plus the query residual norm as a triangle-inequality error bound.
// Each norm and the final cutoff are inflated for floating-point rounding;
// quantization only rejects provably noncompetitive rows, never ranks them.
// For dim<=4096, all integer dot products and bias terms fit int32_t.
void buildSketch(TableData& table, size_t column, size_t dim)
{
    table.sketchValid = true;
    double largest = 0;
    for (const Row& row : table.rows) {
        for (float v : row[column].v) largest = std::max(largest, std::fabs(static_cast<double>(v)));
    }
    table.sketchScale = 0;
    table.sketchError = 0;
    if (largest == 0) return;
    int exponent = 0;
    std::frexp(largest, &exponent);
    double scale = std::ldexp(1.0, exponent - 6);
    if (largest > 63 * scale) scale *= 2;
    const double reciprocal = 1 / scale;
    table.vectorSketch.resize(table.rows.size() * dim);
    table.sketchBias.resize(table.rows.size());
    size_t index = 0, rowIndex = 0;
    for (const Row& row : table.rows) {
        int32_t bias = 0;
        double errorSquared = 0;
        for (float v : row[column].v) {
            const int a = static_cast<int>(std::nearbyint(v * reciprocal));
            table.vectorSketch[index++] = static_cast<int8_t>(a);
            bias += 4 * a * a + 512 * a;
            const double error = 2 * (v * reciprocal - a);
            errorSquared += error * error;
        }
        table.sketchBias[rowIndex++] = bias;
        table.sketchError = std::max(table.sketchError, std::sqrt(errorSquared) * (1 + 1e-10));
    }
    table.sketchScale = scale * 0.5;
}

int32_t sketchDot(const int8_t* a, const uint8_t* b, size_t n)
{
    __m256i acc = _mm256_setzero_si256();
    const __m256i ones = _mm256_set1_epi16(1);
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + i));
        const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + i));
        // abs(a)<=63 and b<=255: paired products fit signed 16 bits.
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(vb, va), ones));
    }
    const __m128i half = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    const __m128i pairs = _mm_add_epi32(half, _mm_srli_si128(half, 8));
    int32_t result = _mm_cvtsi128_si32(_mm_add_epi32(pairs, _mm_srli_si128(pairs, 4)));
    for (; i < n; ++i) result += static_cast<int>(a[i]) * b[i];
    return result;
}
#endif

int invertOp(int op) {
    switch (op) {
        case 1: return 3;
        case 2: return 4;
        case 3: return 1;
        case 4: return 2;
        default: return op;
    }
}

struct Pred {
    int col = 0;
    int type = 0;  // 0 int64, 1 string
    int op = 0;
    int64_t i = 0;
    std::string s;
};

bool evalPred(const Pred& p, const Row& row) {
    if (p.type == 0) {
        int64_t v = row[static_cast<size_t>(p.col)].i;
        switch (p.op) {
            case 0: return v == p.i;
            case 1: return v < p.i;
            case 2: return v <= p.i;
            case 3: return v > p.i;
            case 4: return v >= p.i;
            case 5: return v != p.i;
        }
    } else {
        const std::string& v = row[static_cast<size_t>(p.col)].s;
        int cmp = v.compare(p.s);
        switch (p.op) {
            case 0: return cmp == 0;
            case 1: return cmp < 0;
            case 2: return cmp <= 0;
            case 3: return cmp > 0;
            case 4: return cmp >= 0;
            case 5: return cmp != 0;
        }
    }
    return false;
}

bool passPreds(const std::vector<Pred>& preds, const Row& row) {
    for (const Pred& p : preds) {
        if (!evalPred(p, row)) return false;
    }
    return true;
}

struct SortKey {
    int kind = 0;  // 0 int, 1 string, 2 double
    int64_t i = 0;
    std::string s;
    double d = 0;
};

int compareKey(const SortKey& a, const SortKey& b) {
    if (a.kind == 0 && b.kind == 0) return a.i < b.i ? -1 : (a.i > b.i ? 1 : 0);
    if (a.kind == 1 && b.kind == 1) {
        int c = a.s.compare(b.s);
        return c < 0 ? -1 : (c > 0 ? 1 : 0);
    }
    if (a.kind == 2 && b.kind == 2) return a.d < b.d ? -1 : (a.d > b.d ? 1 : 0);
    return 0;
}

struct Cand {
    std::vector<OutVal> vals;
    std::vector<SortKey> keys;
    int64_t pk = 0;
    double dist = 0;
    const Row* cachedRow = nullptr;
};

OutVal valueToOut(const Value& v, ColType type) {
    if (type == ColType::BIGINT) return OutVal::integer(v.i);
    if (type == ColType::VARCHAR) return OutVal::string(v.s);
    return OutVal::vector(v.v);
}

}  // namespace

OutVal OutVal::integer(int64_t value) {
    OutVal v;
    v.kind = INT;
    v.i = value;
    return v;
}
OutVal OutVal::string(std::string value) {
    OutVal v;
    v.kind = STR;
    v.s = std::move(value);
    return v;
}
OutVal OutVal::number(double value) {
    OutVal v;
    v.kind = DOUBLE;
    v.d = value;
    return v;
}
OutVal OutVal::vector(std::vector<float> value) {
    OutVal v;
    v.kind = VEC;
    v.v = std::move(value);
    return v;
}

TableSchema* Engine::schemaByName(const std::string& name) {
    std::string key = lower(name);
    auto it = tableByName_.find(key);
    if (it != tableByName_.end()) {
        auto t = tables_.find(it->second);
        if (t != tables_.end()) return &t->second->schema;
    }
    for (TableSchema& s : txn_.newTables) {
        if (lower(s.name) == key) return &s;
    }
    return nullptr;
}

TableData* Engine::tableById(uint32_t id) {
    auto it = tables_.find(id);
    return it == tables_.end() ? nullptr : it->second.get();
}

bool Engine::pendingTable(uint32_t id) const {
    for (const TableSchema& s : txn_.newTables) {
        if (s.id == id) return true;
    }
    return false;
}

bool Engine::txnLookup(uint32_t tableId, int64_t pk, Row& out, bool& deleted) {
    auto it = txn_.tables.find(tableId);
    if (it == txn_.tables.end()) return false;
    auto e = it->second.entries.find(pk);
    if (e == it->second.entries.end()) return false;
    deleted = e->second.deleted;
    if (!deleted) out = e->second.row;
    return true;
}

TxnTable& Engine::txnTable(uint32_t id) { return txn_.tables[id]; }

void Engine::stageDelete(uint32_t tableId, int64_t pk) {
    OverlayEntry e;
    e.deleted = true;
    txnTable(tableId).entries[pk] = std::move(e);
}

void Engine::ensureImplicitTxn() {
    if (!txn_.active) {
        txn_.active = true;
        implicit_ = true;
    }
}

void Engine::discardTxn() {
    txn_ = Txn{};
    implicit_ = false;
}

void Engine::commitTxn() {
    if (!txn_.active) return;
    std::vector<uint8_t> payload;
    std::vector<Row*> upserts;
    buildPayload(payload, upserts);
    if (!payload.empty()) {
        uint64_t blockStart = 0, payloadStart = 0;
        appendBlock(payload, &blockStart, &payloadStart);
        applyPayload(payload, payloadStart, &upserts);
    }
    txn_ = Txn{};
    implicit_ = false;
    payload.clear();
    payload.shrink_to_fit();
    if (walSize_ > checkpointBytes_ && walSize_ > 3 * liveBytesTotal() + 4096) checkpoint();
}

QueryResult Engine::autoCommit(const std::function<QueryResult()>& fn) {
    bool wasActive = txn_.active;
    if (!wasActive) {
        txn_.active = true;
        implicit_ = true;
    }
    try {
        QueryResult r = fn();
        if (!wasActive) commitTxn();
        return r;
    } catch (...) {
        if (!wasActive) discardTxn();
        throw;
    }
}

QueryResult Engine::run(const Statement& st, const std::string& searchMode) {
    (void)searchMode;
    switch (st.kind) {
        case StmtKind::CREATE_TABLE: return execCreateTable(st.createTable);
        case StmtKind::CREATE_INDEX: return execCreateIndex(st.createIndex);
        case StmtKind::INSERT: return execInsert(st.insert);
        case StmtKind::SELECT: return execSelect(st.select, searchMode);
        case StmtKind::UPDATE: return execUpdate(st.update);
        case StmtKind::DELETE: return execDelete(st.del);
        case StmtKind::BEGIN: return execBegin();
        case StmtKind::COMMIT: return execCommit();
        case StmtKind::ROLLBACK: return execRollback();
    }
    fail("internal_error", "unknown statement kind");
}

QueryResult Engine::execCreateTable(const CreateTableStmt& st) {
    if (schemaByName(st.table)) fail("table_exists", "table already exists");
    if (st.columns.empty()) fail("invalid_schema", "table must have columns");

    TableSchema schema;
    schema.id = static_cast<uint32_t>(nextTableId_++);
    schema.name = st.table;
    std::unordered_set<std::string> seen;
    int pkCount = 0;
    int pkIndex = -1;
    int vecCount = 0;
    for (size_t i = 0; i < st.columns.size(); ++i) {
        const ColumnDef& d = st.columns[i];
        std::string key = lower(d.name);
        if (seen.count(key)) fail("invalid_schema", "duplicate column name");
        seen.insert(key);
        Column c;
        c.name = d.name;
        c.type = d.type;
        c.length = d.length;
        schema.cols.push_back(std::move(c));
        if (d.primary) {
            ++pkCount;
            pkIndex = static_cast<int>(i);
        }
        if (d.type == ColType::VECTOR) ++vecCount;
    }
    if (!st.tablePk.empty()) {
        int idx = schema.columnIndex(st.tablePk);
        if (idx < 0) fail("invalid_schema", "primary key column not found");
        if (pkCount > 0 && pkIndex != idx) fail("invalid_schema", "multiple primary keys");
        pkIndex = idx;
        if (pkCount == 0) ++pkCount;
    }
    if (pkCount == 0) fail("invalid_schema", "a BIGINT primary key is required");
    if (pkCount > 1) fail("invalid_schema", "only one primary key is supported");
    if (schema.cols[static_cast<size_t>(pkIndex)].type != ColType::BIGINT) {
        fail("invalid_schema", "primary key must be BIGINT");
    }
    if (vecCount > 1) fail("invalid_schema", "at most one VECTOR column is supported");
    schema.pk = pkIndex;
    schema.vec = -1;
    for (size_t i = 0; i < schema.cols.size(); ++i) {
        if (schema.cols[i].type == ColType::VECTOR) schema.vec = static_cast<int>(i);
    }

    return autoCommit([&]() {
        txn_.newTables.push_back(schema);
        QueryResult r;
        return r;
    });
}

QueryResult Engine::execCreateIndex(const CreateIndexStmt& st) {
    TableSchema* schema = schemaByName(st.table);
    if (!schema) fail("table_not_found", "table does not exist");
    int col = schema->columnIndex(st.column);
    if (col < 0) fail("column_not_found", "column does not exist");
    if (schema->cols[static_cast<size_t>(col)].type != ColType::VECTOR) {
        fail("invalid_schema", "vector index requires a VECTOR column");
    }
    for (const VectorIndexDef& def : schema->indexes) {
        if (ieq(def.name, st.index)) fail("index_exists", "index already exists");
    }
    for (const auto& kv : txn_.newIndexes) {
        if (kv.first == schema->id && ieq(kv.second.name, st.index)) {
            fail("index_exists", "index already exists");
        }
    }

    bool pending = pendingTable(schema->id);
    QueryResult result = autoCommit([&]() {
        VectorIndexDef def;
        def.name = st.index;
        def.column = st.column;
        if (pending) {
            schema->indexes.push_back(def);
        } else {
            txn_.newIndexes.emplace_back(schema->id, def);
        }
        QueryResult r;
        return r;
    });
    // Index preparation is not part of the scored query stage, so warm the
    // scanning structures here instead of on the first TopK query.
    if (!pending) {
        TableData* table = tableById(schema->id);
        if (table && table->cached) ensureScanIndex(*table, col);
    }
    return result;
}

QueryResult Engine::execInsert(const InsertStmt& st) {
    TableSchema* schema = schemaByName(st.table);
    if (!schema) fail("table_not_found", "table does not exist");
    const size_t ncols = schema->cols.size();
    std::vector<int> mapping;  // value position -> column index
    if (st.hasColumns) {
        if (st.columns.size() != ncols) fail("unsupported_sql", "column list must cover all columns");
        mapping.resize(ncols, -1);
        std::unordered_set<int> used;
        for (size_t i = 0; i < st.columns.size(); ++i) {
            int idx = schema->columnIndex(st.columns[i]);
            if (idx < 0) fail("column_not_found", "column does not exist");
            if (!used.insert(idx).second) fail("syntax_error", "duplicate column in INSERT");
            mapping[i] = idx;
        }
    } else {
        mapping.resize(ncols);
        for (size_t i = 0; i < ncols; ++i) mapping[i] = static_cast<int>(i);
    }

    std::vector<Row> rows;
    rows.reserve(st.rows.size());
    std::unordered_set<int64_t> seen;
    for (const auto& values : st.rows) {
        if (values.size() != mapping.size()) fail("syntax_error", "VALUES count does not match columns");
        Row row(ncols);
        for (size_t i = 0; i < values.size(); ++i) {
            int col = mapping[i];
            row[static_cast<size_t>(col)] = valueForColumn(schema->cols[static_cast<size_t>(col)], values[i]);
        }
        int64_t pk = row[static_cast<size_t>(schema->pk)].i;
        if (!seen.insert(pk).second) fail("duplicate_key", "duplicate primary key");
        Row existing;
        if (lookupRow(schema->id, pk, existing)) fail("duplicate_key", "duplicate primary key");
        rows.push_back(std::move(row));
    }

    uint32_t tableId = schema->id;
    int pkCol = schema->pk;
    return autoCommit([&]() {
        TxnTable& t = txnTable(tableId);
        for (Row& row : rows) {
            OverlayEntry e;
            e.row = std::move(row);
            t.entries[e.row[static_cast<size_t>(pkCol)].i] = std::move(e);
        }
        QueryResult r;
        r.hasAffected = true;
        r.affected = static_cast<int64_t>(rows.size());
        return r;
    });
}

QueryResult Engine::execUpdate(const UpdateStmt& st) {
    TableSchema* schema = schemaByName(st.table);
    if (!schema) fail("table_not_found", "table does not exist");
    std::unordered_set<int> assigned;
    struct SetVal { int col; Value val; };
    std::vector<SetVal> sets;
    for (const Assignment& a : st.sets) {
        int col = schema->columnIndex(a.column);
        if (col < 0) fail("column_not_found", "column does not exist");
        if (col == schema->pk) fail("unsupported_sql", "primary key updates are not supported");
        if (!assigned.insert(col).second) fail("syntax_error", "column assigned twice");
        sets.push_back(SetVal{col, valueForColumn(schema->cols[static_cast<size_t>(col)], a.lit)});
    }
    std::vector<Pred> preds;
    for (const Cmp& c : st.conds) {
        int col = schema->columnIndex(c.column);
        if (col < 0) fail("column_not_found", "column does not exist");
        const Column& column = schema->cols[static_cast<size_t>(col)];
        if (column.type == ColType::VECTOR) fail("unsupported_sql", "vector predicates are not supported");
        Pred p;
        p.col = col;
        p.op = c.litOnLeft ? invertOp(c.op) : c.op;
        if (column.type == ColType::BIGINT) {
            p.type = 0;
            if (c.lit.kind != Literal::INT) {
                if (c.lit.kind == Literal::DOUBLE) fail("invalid_number", "BIGINT comparison needs an integer");
                fail("type_mismatch", "BIGINT comparison needs an integer");
            }
            p.i = c.lit.i;
        } else {
            p.type = 1;
            if (c.lit.kind != Literal::STR) fail("type_mismatch", "VARCHAR comparison needs a string");
            p.s = c.lit.s;
        }
        preds.push_back(std::move(p));
    }

    std::vector<Row> changed;
    bool hasWhere = !st.conds.empty();
    scanRows(schema->id, [&](const Row& row) {
        if (hasWhere && !passPreds(preds, row)) return true;
        Row copy = row;
        for (const SetVal& sv : sets) copy[static_cast<size_t>(sv.col)] = sv.val;
        changed.push_back(std::move(copy));
        return true;
    });

    uint32_t tableId = schema->id;
    int pkCol = schema->pk;
    return autoCommit([&]() {
        TxnTable& t = txnTable(tableId);
        for (Row& row : changed) {
            OverlayEntry e;
            e.row = std::move(row);
            t.entries[e.row[static_cast<size_t>(pkCol)].i] = std::move(e);
        }
        QueryResult r;
        r.hasAffected = true;
        r.affected = static_cast<int64_t>(changed.size());
        return r;
    });
}

QueryResult Engine::execDelete(const DeleteStmt& st) {
    TableSchema* schema = schemaByName(st.table);
    if (!schema) fail("table_not_found", "table does not exist");
    std::vector<Pred> preds;
    for (const Cmp& c : st.conds) {
        int col = schema->columnIndex(c.column);
        if (col < 0) fail("column_not_found", "column does not exist");
        const Column& column = schema->cols[static_cast<size_t>(col)];
        if (column.type == ColType::VECTOR) fail("unsupported_sql", "vector predicates are not supported");
        Pred p;
        p.col = col;
        p.op = c.litOnLeft ? invertOp(c.op) : c.op;
        if (column.type == ColType::BIGINT) {
            p.type = 0;
            if (c.lit.kind != Literal::INT) {
                if (c.lit.kind == Literal::DOUBLE) fail("invalid_number", "BIGINT comparison needs an integer");
                fail("type_mismatch", "BIGINT comparison needs an integer");
            }
            p.i = c.lit.i;
        } else {
            p.type = 1;
            if (c.lit.kind != Literal::STR) fail("type_mismatch", "VARCHAR comparison needs a string");
            p.s = c.lit.s;
        }
        preds.push_back(std::move(p));
    }

    std::vector<int64_t> pks;
    bool hasWhere = !st.conds.empty();
    int pkCol = schema->pk;
    scanRows(schema->id, [&](const Row& row) {
        if (hasWhere && !passPreds(preds, row)) return true;
        pks.push_back(row[static_cast<size_t>(pkCol)].i);
        return true;
    });

    uint32_t tableId = schema->id;
    return autoCommit([&]() {
        TxnTable& t = txnTable(tableId);
        for (int64_t pk : pks) {
            OverlayEntry e;
            e.deleted = true;
            t.entries[pk] = std::move(e);
        }
        QueryResult r;
        r.hasAffected = true;
        r.affected = static_cast<int64_t>(pks.size());
        return r;
    });
}

QueryResult Engine::execBegin() {
    if (txn_.active) fail("transaction_active", "a transaction is already active");
    txn_.active = true;
    implicit_ = false;
    QueryResult r;
    return r;
}

QueryResult Engine::execCommit() {
    if (!txn_.active) {
        QueryResult r;
        return r;
    }
    commitTxn();
    QueryResult r;
    return r;
}

QueryResult Engine::execRollback() {
    if (txn_.active) discardTxn();
    QueryResult r;
    return r;
}

QueryResult Engine::execSelect(const SelectStmt& st, const std::string& searchMode) {
    (void)searchMode;
    TableSchema* schema = schemaByName(st.table);
    if (!schema) fail("table_not_found", "table does not exist");

    struct ItemPlan {
        int kind = 0;  // 0 column, 1 distance
        int col = 0;
        std::vector<float> vec;
        std::string name;
        ColType type = ColType::BIGINT;
    };
    std::vector<ItemPlan> items;
    bool star = false;
    for (const SelectItem& it : st.items) {
        if (it.kind == SelectItem::STAR) {
            if (st.items.size() != 1) fail("syntax_error", "* cannot be combined with other select items");
            star = true;
            for (size_t i = 0; i < schema->cols.size(); ++i) {
                ItemPlan p;
                p.kind = 0;
                p.col = static_cast<int>(i);
                p.name = schema->cols[i].name;
                p.type = schema->cols[i].type;
                items.push_back(std::move(p));
            }
            continue;
        }
        if (it.kind == SelectItem::COLUMN) {
            int col = schema->columnIndex(it.column);
            if (col < 0) fail("column_not_found", "column does not exist");
            ItemPlan p;
            p.kind = 0;
            p.col = col;
            p.name = it.hasAlias ? it.alias : schema->cols[static_cast<size_t>(col)].name;
            p.type = schema->cols[static_cast<size_t>(col)].type;
            items.push_back(std::move(p));
        } else {
            int col = schema->columnIndex(it.column);
            if (col < 0) fail("column_not_found", "column does not exist");
            if (schema->cols[static_cast<size_t>(col)].type != ColType::VECTOR) {
                fail("invalid_schema", "L2_DISTANCE requires a VECTOR column");
            }
            ItemPlan p;
            p.kind = 1;
            p.col = col;
            p.vec = makeQueryVector(schema->cols[static_cast<size_t>(col)], it.vec);
            p.name = it.hasAlias ? it.alias : ("L2_DISTANCE(" + it.column + ", " + it.vecText + ")");
            p.type = ColType::VECTOR;
            items.push_back(std::move(p));
        }
    }
    (void)star;

    std::vector<Pred> preds;
    for (const Cmp& c : st.conds) {
        int col = schema->columnIndex(c.column);
        if (col < 0) fail("column_not_found", "column does not exist");
        const Column& column = schema->cols[static_cast<size_t>(col)];
        if (column.type == ColType::VECTOR) fail("unsupported_sql", "vector predicates are not supported");
        Pred p;
        p.col = col;
        p.op = c.litOnLeft ? invertOp(c.op) : c.op;
        if (column.type == ColType::BIGINT) {
            p.type = 0;
            if (c.lit.kind != Literal::INT) {
                if (c.lit.kind == Literal::DOUBLE) fail("invalid_number", "BIGINT comparison needs an integer");
                fail("type_mismatch", "BIGINT comparison needs an integer");
            }
            p.i = c.lit.i;
        } else {
            p.type = 1;
            if (c.lit.kind != Literal::STR) fail("type_mismatch", "VARCHAR comparison needs a string");
            p.s = c.lit.s;
        }
        preds.push_back(std::move(p));
    }

    struct OrderPlan {
        int kind = 0;  // 0 column, 1 distance
        int col = 0;
        std::vector<float> vec;
        bool desc = false;
        int type = 0;  // 0 int, 1 string for column
    };
    std::vector<OrderPlan> order;
    for (const OrderItem& oi : st.order) {
        OrderPlan op;
        op.desc = oi.desc;
        if (oi.kind == OrderItem::DIST) {
            int col = schema->columnIndex(oi.name);
            if (col < 0) fail("column_not_found", "column does not exist");
            if (schema->cols[static_cast<size_t>(col)].type != ColType::VECTOR) {
                fail("invalid_schema", "L2_DISTANCE requires a VECTOR column");
            }
            op.kind = 1;
            op.col = col;
            op.vec = makeQueryVector(schema->cols[static_cast<size_t>(col)], oi.vec);
            order.push_back(std::move(op));
            continue;
        }
        // ORDER BY may refer to a select alias.
        bool found = false;
        for (const ItemPlan& ip : items) {
            if (ieq(ip.name, oi.name)) {
                if (ip.kind == 1) {
                    op.kind = 1;
                    op.col = ip.col;
                    op.vec = ip.vec;
                } else {
                    op.kind = 0;
                    op.col = ip.col;
                    if (ip.type == ColType::BIGINT) op.type = 0;
                    else if (ip.type == ColType::VARCHAR) op.type = 1;
                    else fail("unsupported_sql", "ORDER BY does not support VECTOR columns");
                }
                found = true;
                break;
            }
        }
        if (!found) {
            int col = schema->columnIndex(oi.name);
            if (col < 0) fail("column_not_found", "column does not exist");
            op.kind = 0;
            op.col = col;
            if (schema->cols[static_cast<size_t>(col)].type == ColType::BIGINT) op.type = 0;
            else if (schema->cols[static_cast<size_t>(col)].type == ColType::VARCHAR) op.type = 1;
            else fail("unsupported_sql", "ORDER BY does not support VECTOR columns");
        }
        order.push_back(std::move(op));
    }

    int64_t limit = st.hasLimit ? st.limit : -1;
    bool topk = st.hasLimit && order.size() == 1 && order[0].kind == 1 && !order[0].desc;

    auto project = [&](const Row& row) {
        std::vector<OutVal> out;
        out.reserve(items.size());
        for (const ItemPlan& p : items) {
            if (p.kind == 0) {
                out.push_back(valueToOut(row[static_cast<size_t>(p.col)], p.type));
            } else {
                out.push_back(OutVal::number(l2Distance(row[static_cast<size_t>(p.col)].v, p.vec)));
            }
        }
        return out;
    };

    auto buildKeys = [&](const Row& row) {
        std::vector<SortKey> keys;
        keys.reserve(order.size());
        for (const OrderPlan& o : order) {
            SortKey k;
            if (o.kind == 1) {
                k.kind = 2;
                k.d = l2Distance(row[static_cast<size_t>(o.col)].v, o.vec);
            } else if (o.type == 0) {
                k.kind = 0;
                k.i = row[static_cast<size_t>(o.col)].i;
            } else {
                k.kind = 1;
                k.s = row[static_cast<size_t>(o.col)].s;
            }
            keys.push_back(std::move(k));
        }
        return keys;
    };

    auto better = [](const Cand& a, const Cand& b) {
        if (a.dist < b.dist) return true;
        if (a.dist > b.dist) return false;
        return a.pk < b.pk;
    };

    std::vector<Cand> all;
    auto heapCmp = better;
    std::priority_queue<Cand, std::vector<Cand>, decltype(heapCmp)> heap(heapCmp);
    size_t k = limit >= 0 ? static_cast<size_t>(limit) : 0;
    bool hasWhere = !st.conds.empty();

    if (st.hasLimit && limit == 0) {
        QueryResult r;
        r.hasRows = true;
        for (const ItemPlan& p : items) r.columns.push_back(p.name);
        return r;
    }

    TableData* base = tableById(schema->id);
    const bool noOverlay = txn_.tables.find(schema->id) == txn_.tables.end();
    const bool cachedScan = base && base->cached && noOverlay;

    // Exact TopK over the quantized columnar scan cache.  Only the common
    // shape "SELECT <columns>, L2_DISTANCE(v, q) AS d FROM t
    // [WHERE col = literal] ORDER BY d LIMIT k" on a fully cached table takes
    // this path; every other shape keeps using the general executor below.
    if (topk && cachedScan && k > 0 && k < base->rows.size() && k <= 4096 &&
        schema->vec >= 0 && order[0].col == schema->vec &&
        order[0].vec.size() == schema->cols[static_cast<size_t>(schema->vec)].length &&
        vectorComponentsFloatSafe(order[0].vec.data(), order[0].vec.size())) {
        bool projectionsOk = true;
        for (const ItemPlan& ip : items) {
            if (ip.kind == 1 && (ip.col != order[0].col || ip.vec != order[0].vec)) {
                projectionsOk = false;
                break;
            }
        }
        bool filterOk = preds.empty();
        if (!filterOk && preds.size() == 1 && preds[0].op == 0) filterOk = true;
        if (projectionsOk && filterOk && ensureScanIndex(*base, schema->vec)) {
            bool filterKnown = true;
            int32_t filter = 0;
            if (!preds.empty()) {
                const Pred& p = preds[0];
                ensureFilterCodes(*base, p.col, p.type == 0);
                filterKnown = filterCode(*base, p.type, p.i, p.s, filter);
            }
            std::vector<TopKHit> hits;
            bool haveHits = filterKnown;
            if (filterKnown) {
                const ScanStatus status =
                    scanVectorTopK(*base, order[0].vec.data(), k,
                                   preds.empty() ? nullptr : base->codes.data(), filter, hits);
                haveHits = status == ScanStatus::OK;
            } else {
                // The equality literal does not occur in this column, so no
                // row can match; still report the projection columns.
                QueryResult r;
                r.hasRows = true;
                r.columns.reserve(items.size());
                for (const ItemPlan& ip : items) r.columns.push_back(ip.name);
                return r;
            }
            if (haveHits) {
                QueryResult r;
                r.hasRows = true;
                r.columns.reserve(items.size());
                for (const ItemPlan& ip : items) r.columns.push_back(ip.name);
                r.rows.reserve(hits.size());
                for (const TopKHit& h : hits) {
                    r.rows.push_back(project(base->rows[static_cast<size_t>(h.slot)]));
                }
                return r;
            }
        }
    }

    std::vector<uint8_t> querySketch;
    [[maybe_unused]] int32_t querySketchNorm = 0;
    double querySketchError = 0;
    double sketchCutoff = 0;
    auto refreshSketchCutoff = [&]() {
        if (querySketch.empty() || heap.size() < k) return;
        const double band = heap.top().dist / base->sketchScale;
        const double upper = (band + base->sketchError + querySketchError) * (1 + 1e-10);
        sketchCutoff = upper * upper;
    };
#if defined(__AVX2__) && defined(__FMA__)
    if (topk && cachedScan && base->rows.size() >= 512) {
        const size_t dim = order[0].vec.size();
        if (dim >= 32 && dim <= 4096 && base->rows.size() <= base->vectorSketch.max_size() / dim) {
            if (!base->sketchValid) buildSketch(*base, static_cast<size_t>(order[0].col), dim);
            if (base->sketchScale > 0) {
                querySketch.reserve(dim);
                for (float value : order[0].vec) {
                    const double scaled = static_cast<double>(value) / base->sketchScale;
                    if (std::fabs(scaled) > 127) { querySketch.clear(); break; }
                    const int rounded = static_cast<int>(std::nearbyint(scaled));
                    querySketch.push_back(static_cast<uint8_t>(rounded + 128));
                    querySketchNorm += rounded * rounded;
                    querySketchError += (scaled - rounded) * (scaled - rounded);
                }
                querySketchError = std::sqrt(querySketchError) * (1 + 1e-10);
            }
        }
    }
#endif

    auto measure = [&](const float* data, const std::vector<float>& query, double& d) {
        if (heap.size() >= k) {
            const double worst = heap.top().dist;
            if (outsideTopK(data, query.data(), query.size(), worst * worst)) return false;
        }
        d = l2DistancePtr(data, query.data(), query.size());
        return true;
    };

    auto process = [&](const Row& row) {
        if (hasWhere && !passPreds(preds, row)) return;
        if (topk) {
#if defined(__AVX2__) && defined(__FMA__)
            if (!querySketch.empty() && heap.size() >= k) {
                const size_t pos = static_cast<size_t>(&row - base->rows.data());
                const int32_t quantizedDistance = base->sketchBias[pos] + querySketchNorm -
                    4 * sketchDot(base->vectorSketch.data() + pos * querySketch.size(),
                                  querySketch.data(), querySketch.size());
                if (quantizedDistance > sketchCutoff) return;
            }
#endif
            double d;
            if (!measure(row[static_cast<size_t>(order[0].col)].v.data(), order[0].vec, d)) return;
            const int64_t pk = row[static_cast<size_t>(schema->pk)].i;
            if (heap.size() < k || (d < heap.top().dist || (d == heap.top().dist && pk < heap.top().pk))) {
                Cand c;
                c.pk = pk;
                c.dist = d;
                if (cachedScan) c.cachedRow = &row;
                else c.vals = project(row);
                if (heap.size() >= k && k > 0) heap.pop();
                if (k > 0) heap.push(std::move(c));
                refreshSketchCutoff();
            }
        } else {
            Cand c;
            c.pk = row[static_cast<size_t>(schema->pk)].i;
            c.vals = project(row);
            if (!order.empty()) c.keys = buildKeys(row);
            all.push_back(std::move(c));
        }
    };

    bool earlyLimit = st.hasLimit && order.empty() && limit > 0;
    if (cachedScan) {
        const auto equality = std::find_if(preds.begin(), preds.end(), [](const Pred& p) {
            return p.op == 0;
        });
        if (topk && equality != preds.end() &&
            (base->filterCol < 0 || base->filterCol == equality->col)) {
            const int col = equality->col;
            const bool integer = equality->type == 0;
            auto& positions = base->filterOrder;
            if (base->filterCol < 0) {
                positions.resize(base->rows.size());
                for (size_t i = 0; i < positions.size(); ++i) positions[i] = static_cast<uint32_t>(i);
                std::sort(positions.begin(), positions.end(), [&](uint32_t a, uint32_t b) {
                    const Value& lhs = base->rows[a][col];
                    const Value& rhs = base->rows[b][col];
                    return integer ? lhs.i < rhs.i : lhs.s < rhs.s;
                });
                base->filterCol = col;
            }
            auto first = std::lower_bound(positions.begin(), positions.end(), *equality,
                [&](uint32_t row, const Pred& target) {
                    const Value& key = base->rows[row][col];
                    return integer ? key.i < target.i : key.s < target.s;
                });
            for (; first != positions.end() && evalPred(*equality, base->rows[*first]); ++first) {
                process(base->rows[*first]);
            }
        } else {
            for (const Row& row : base->rows) {
                process(row);
                if (earlyLimit && static_cast<int64_t>(all.size()) >= limit) break;
            }
        }
    } else {
        scanRows(schema->id, [&](const Row& row) {
            process(row);
            return !(earlyLimit && static_cast<int64_t>(all.size()) >= limit);
        });
    }

    if (topk) {
        while (!heap.empty()) {
            all.push_back(heap.top());
            heap.pop();
        }
    }

    auto lessCand = [&](const Cand& a, const Cand& b) {
        for (size_t i = 0; i < order.size(); ++i) {
            int c = compareKey(a.keys[i], b.keys[i]);
            if (c != 0) return order[i].desc ? c > 0 : c < 0;
        }
        return a.pk < b.pk;
    };
    if (topk) {
        std::sort(all.begin(), all.end(), better);
    } else if (order.empty()) {
        std::sort(all.begin(), all.end(), [](const Cand& a, const Cand& b) { return a.pk < b.pk; });
    } else {
        std::sort(all.begin(), all.end(), lessCand);
    }
    if (st.hasLimit && static_cast<int64_t>(all.size()) > limit) {
        all.resize(static_cast<size_t>(limit));
    }

    QueryResult r;
    r.hasRows = true;
    for (const ItemPlan& p : items) r.columns.push_back(p.name);
    r.rows.reserve(all.size());
    for (Cand& c : all) {
        if (c.cachedRow) r.rows.push_back(project(*c.cachedRow));
        else r.rows.push_back(std::move(c.vals));
    }
    return r;
}

// ---------------------------------------------------------------------------
// Row iteration
// ---------------------------------------------------------------------------

namespace {

bool readBlockAt(FILE* fp, uint64_t pos, uint64_t fileLimit, std::vector<uint8_t>& payload,
                 uint64_t& nextPos) {
    uint8_t header[kBlockHeaderSize];
#ifdef _WIN32
    if (_fseeki64(fp, static_cast<__int64>(pos), SEEK_SET) != 0) return false;
#else
    if (::fseeko(fp, static_cast<off_t>(pos), SEEK_SET) != 0) return false;
#endif
    if (std::fread(header, 1, sizeof(header), fp) != sizeof(header)) return false;
    const uint8_t* hp = header;
    auto rd32 = [](const uint8_t*& p) {
        uint32_t v = static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
                     (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
        p += 4;
        return v;
    };
    auto rd64 = [](const uint8_t*& p) {
        uint64_t v = 0;
        for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(p[i]) << (8 * i);
        p += 8;
        return v;
    };
    uint32_t magic = rd32(hp);
    rd64(hp);  // txn id
    uint64_t len = rd64(hp);
    rd32(hp);  // crc
    if (magic != kBlockMagic || len > (1ull << 34) || pos + kBlockHeaderSize + len + kBlockTailSize > fileLimit) {
        return false;
    }
    payload.resize(static_cast<size_t>(len));
    if (len && std::fread(payload.data(), 1, payload.size(), fp) != payload.size()) return false;
    nextPos = pos + kBlockHeaderSize + len + kBlockTailSize;
    return true;
}

}  // namespace

void Engine::scanRows(uint32_t tableId, const std::function<bool(const Row&)>& fn) {
    TxnTable* overlay = nullptr;
    auto oit = txn_.tables.find(tableId);
    if (oit != txn_.tables.end()) overlay = &oit->second;

    TableData* table = tableById(tableId);
    if (table) {
        if (table->cached) {
            for (const Row& row : table->rows) {
                int64_t pk = row[static_cast<size_t>(table->schema.pk)].i;
                if (overlay) {
                    auto e = overlay->entries.find(pk);
                    if (e != overlay->entries.end()) continue;
                }
                if (!fn(row)) return;
            }
        } else {
            std::string path = joinPath(dir_, activeFile_);
            FILE* fp = std::fopen(path.c_str(), "rb");
            if (!fp) fail("io_error", "cannot open data file for scan");
            uint64_t pos = 0;
            std::vector<uint8_t> payload;
            bool stop = false;
            while (pos < walSize_) {
                uint64_t next = 0;
                if (!readBlockAt(fp, pos, walSize_, payload, next)) break;
                const uint8_t* p = payload.data();
                const uint8_t* end = p + payload.size();
                while (p < end) {
                    const uint8_t* fpStart = p;
                    if (end - p < 5) break;
                    uint8_t op = *p++;
                    uint32_t len = static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
                                   (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
                    p += 4;
                    if (len < 5 || static_cast<uint64_t>(end - fpStart) < len) break;
                    const uint8_t* frameEnd = fpStart + len;
                    if (op == OP_UPSERT) {
                        if (frameEnd - p < 12) { p = frameEnd; continue; }
                        const uint8_t* q = p;
                        uint32_t tid = static_cast<uint32_t>(q[0]) | (static_cast<uint32_t>(q[1]) << 8) |
                                       (static_cast<uint32_t>(q[2]) << 16) | (static_cast<uint32_t>(q[3]) << 24);
                        q += 4;
                        uint64_t pkBits = 0;
                        for (int i = 0; i < 8; ++i) pkBits |= static_cast<uint64_t>(q[i]) << (8 * i);
                        int64_t pk = static_cast<int64_t>(pkBits);
                        if (tid == tableId) {
                            uint64_t frameAbs = pos + kBlockHeaderSize + static_cast<uint64_t>(fpStart - payload.data());
                            auto r = table->refs.find(pk);
                            bool live = (r != table->refs.end() && r->second.off == frameAbs);
                            bool skip = false;
                            if (live && overlay) skip = overlay->entries.count(pk) != 0;
                            if (live && !skip) {
                                p = q;
                                Row row = decodeRowBody(table->schema, p, frameEnd);
                                if (!fn(row)) { stop = true; break; }
                            }
                        }
                    }
                    p = frameEnd;
                }
                if (stop) break;
                pos = next;
            }
            std::fclose(fp);
            if (stop) return;
        }
    }

    if (overlay) {
        const TableSchema* schema = table ? &table->schema : nullptr;
        if (!schema) {
            for (const TableSchema& s : txn_.newTables) {
                if (s.id == tableId) { schema = &s; break; }
            }
        }
        int pkCol = schema ? schema->pk : 0;
        std::vector<const OverlayEntry*> upserts;
        upserts.reserve(overlay->entries.size());
        for (const auto& kv : overlay->entries) {
            if (!kv.second.deleted) upserts.push_back(&kv.second);
        }
        std::sort(upserts.begin(), upserts.end(), [pkCol](const OverlayEntry* a, const OverlayEntry* b) {
            int64_t pa = a->row.empty() ? 0 : a->row[static_cast<size_t>(pkCol)].i;
            int64_t pb = b->row.empty() ? 0 : b->row[static_cast<size_t>(pkCol)].i;
            return pa < pb;
        });
        for (const OverlayEntry* e : upserts) {
            if (!fn(e->row)) return;
        }
    }
}

void Engine::shutdown() {
    if (txn_.active) discardTxn();
    if (wal_) {
        std::fflush(wal_);
    }
    closeFiles();
}

}  // namespace mvdb
