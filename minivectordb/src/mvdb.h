// MiniVectorDB - a small single-process vector database.
//
// The design is deliberately simple: every committed change is appended to a
// single redo file as one self-checking transaction block, and the latest row
// image of every primary key is indexed in memory.  Small tables keep decoded
// rows in a cache; large tables are scanned from the file so the process does
// not need the whole data set in RAM.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace mvdb {

enum class ColType : uint8_t { BIGINT = 1, VARCHAR = 2, VECTOR = 3 };

struct Column {
    std::string name;
    ColType type = ColType::BIGINT;
    uint32_t length = 0;  // VARCHAR character limit or VECTOR dimension
};

struct VectorIndexDef {
    std::string name;
    std::string column;
};

struct TableSchema {
    uint32_t id = 0;
    std::string name;
    std::vector<Column> cols;
    int pk = -1;
    int vec = -1;
    std::vector<VectorIndexDef> indexes;

    int columnIndex(const std::string& columnName) const;
    int vectorColumnCount() const;
};

struct Value {
    int64_t i = 0;
    std::string s;
    std::vector<float> v;
};

using Row = std::vector<Value>;

struct DbError : public std::runtime_error {
    std::string code;
    DbError(std::string c, const std::string& message)
        : std::runtime_error(message), code(std::move(c)) {}
};

[[noreturn]] void fail(const std::string& code, const std::string& message);

// A value in a result set.  Keeping the JSON type explicit matters because the
// judge compares integers, strings and floats with different rules.
struct OutVal {
    enum Kind { INT, STR, DOUBLE, VEC } kind = INT;
    int64_t i = 0;
    std::string s;
    double d = 0;
    std::vector<float> v;

    static OutVal integer(int64_t value);
    static OutVal string(std::string value);
    static OutVal number(double value);
    static OutVal vector(std::vector<float> value);
};

struct QueryResult {
    bool hasRows = false;
    std::vector<std::string> columns;
    std::vector<std::vector<OutVal>> rows;
    bool hasAffected = false;
    int64_t affected = 0;
};

// ---------------------------------------------------------------------------
// Parsed SQL
// ---------------------------------------------------------------------------

struct Literal {
    enum Kind { INT, DOUBLE, STR, VECLIT, NULLV } kind = INT;
    int64_t i = 0;
    double d = 0;
    std::string s;
    std::vector<double> v;
};

struct ColumnDef {
    std::string name;
    ColType type = ColType::BIGINT;
    uint32_t length = 0;
    bool primary = false;
};

struct CreateTableStmt {
    std::string table;
    std::vector<ColumnDef> columns;
    std::string tablePk;  // table-level PRIMARY KEY (col)
};

struct CreateIndexStmt {
    std::string index;
    std::string table;
    std::string column;
};

struct SelectItem {
    enum Kind { STAR, COLUMN, DIST } kind = COLUMN;
    std::string column;
    std::string alias;
    bool hasAlias = false;
    std::vector<double> vec;  // DIST argument
    std::string vecText;
};

struct OrderItem {
    enum Kind { COLUMN, DIST } kind = COLUMN;
    std::string name;
    bool desc = false;
    std::vector<double> vec;
    std::string vecText;
};

struct Cmp {
    std::string column;
    int op = 0;  // 0 '=', 1 '<', 2 '<=', 3 '>', 4 '>=', 5 '<>'
    Literal lit;
    bool litOnLeft = false;
};

struct SelectStmt {
    std::vector<SelectItem> items;
    std::string table;
    std::vector<Cmp> conds;
    std::vector<OrderItem> order;
    bool hasLimit = false;
    int64_t limit = 0;
};

struct InsertStmt {
    std::string table;
    std::vector<std::string> columns;
    bool hasColumns = false;
    std::vector<std::vector<Literal>> rows;
};

struct Assignment {
    std::string column;
    Literal lit;
};

struct UpdateStmt {
    std::string table;
    std::vector<Assignment> sets;
    std::vector<Cmp> conds;
    bool hasWhere = false;
};

struct DeleteStmt {
    std::string table;
    std::vector<Cmp> conds;
    bool hasWhere = false;
};

enum class StmtKind {
    CREATE_TABLE,
    CREATE_INDEX,
    INSERT,
    SELECT,
    UPDATE,
    DELETE,
    BEGIN,
    COMMIT,
    ROLLBACK
};

struct Statement {
    StmtKind kind = StmtKind::SELECT;
    CreateTableStmt createTable;
    CreateIndexStmt createIndex;
    InsertStmt insert;
    SelectStmt select;
    UpdateStmt update;
    DeleteStmt del;
};

Statement parseStatement(const std::string& sql);
std::vector<double> parseVectorText(const std::string& text, size_t expectedSize = 0);

// ---------------------------------------------------------------------------
// Storage / execution engine
// ---------------------------------------------------------------------------

constexpr uint32_t kBlockMagic = 0x4D564231u;  // "MVB1"
constexpr uint32_t kTailMagic = 0x4D564531u;   // "MVE1"
constexpr size_t kBlockHeaderSize = 28;
constexpr size_t kBlockTailSize = 24;

enum : uint8_t {
    OP_CREATE_TABLE = 1,
    OP_UPSERT = 2,
    OP_DELETE = 3,
    OP_CREATE_INDEX = 4
};

struct DiskRef {
    uint64_t off = 0;
    uint32_t size = 0;  // encoded row body size, used for garbage accounting
};

struct TableData {
    TableSchema schema;
    bool cached = false;
    std::unordered_map<int64_t, int> slotOf;
    std::vector<Row> rows;
    std::unordered_map<int64_t, DiskRef> refs;
    uint64_t liveBytes = 0;
    uint64_t cachedBytes = 0;
    int filterCol = -1;
    std::vector<uint32_t> filterOrder;
    bool sketchValid = false;
    double sketchScale = 0;
    double sketchError = 0;
    std::vector<int8_t> vectorSketch;
    std::vector<int32_t> sketchBias;

    // TopK scan cache for the single VECTOR column.  Rows keep the
    // authoritative values; this cache mirrors them in slot order as 7-bit
    // codes plus a contiguous float32 copy, so a scan reads 128 bytes per row
    // instead of chasing one allocation per row.  It is built lazily on the
    // first exact TopK query and maintained by the storage layer, so a write
    // never forces a rebuild.
    bool scanValid = false;
    bool scanSafe = true;  // values fit the float32 rescore and 7-bit codes
    int scanVec = -1;
    uint32_t scanDim = 0;
    float scanScale = 1.0f;         // value = scale * code
    std::vector<uint8_t> scanQ;     // rows * scanDim, code + 63
    std::vector<int32_t> scanNorm;  // sum of squared codes
    std::vector<int64_t> scanPk;    // primary key per slot
    std::vector<float> scanVecs;    // contiguous float32 copy for rescoring
    // Equality codes for one filter column, parallel to the row slots.
    int codeCol = -1;
    bool codeIsInt = true;
    std::vector<int32_t> codes;
    std::unordered_map<int64_t, int32_t> intCodes;
    std::unordered_map<std::string, int32_t> strCodes;
};

struct TopKHit {
    uint32_t slot = 0;
    double dist = 0;
};

enum class ScanStatus {
    OK,          // hits are the exact answer
    UNAVAILABLE, // no usable scan cache for this table/query
    NEED_EXACT   // the shortlist could not be proven, use the exact executor
};

bool vectorComponentsFloatSafe(const float* v, size_t n);
ScanStatus scanVectorTopK(const TableData& table, const float* query, size_t k,
                          const int32_t* codes, int32_t code, std::vector<TopKHit>& out);

struct OverlayEntry {
    bool deleted = false;
    Row row;
};

struct TxnTable {
    std::unordered_map<int64_t, OverlayEntry> entries;
};

struct Txn {
    bool active = false;
    std::unordered_map<uint32_t, TxnTable> tables;
    std::vector<TableSchema> newTables;
    std::vector<std::pair<uint32_t, VectorIndexDef>> newIndexes;
};

class Engine {
public:
    Engine(std::string dataDir, uint64_t cacheBudget);
    ~Engine();

    void recover();
    QueryResult run(const Statement& st, const std::string& searchMode);
    void shutdown();

    const std::string& dataDir() const { return dir_; }
    uint64_t cacheBudget() const { return cacheBudget_; }

private:
    // catalog / files
    void initStorage();
    void recoverStorage();
    void writeCatalogAtomic();
    void cleanupOrphanFiles();
    void openActiveFile();
    void reopenReadFile();
    void closeFiles();
    FILE* openRead();
    void appendBlock(const std::vector<uint8_t>& payload, uint64_t* blockStart,
                     uint64_t* payloadStart);
    void truncateWal(uint64_t size);
    void checkpoint();
    uint64_t liveBytesTotal() const;

    // payload encoding / replay
    void buildPayload(std::vector<uint8_t>& payload, std::vector<Row*>& upserts);
    void applyPayload(const std::vector<uint8_t>& payload, uint64_t payloadOffset,
                      const std::vector<Row*>* committedRows = nullptr);
    void applyUpsertRow(uint32_t tableId, Row& row, uint64_t off, uint32_t bodySize);
    void applyUpsertRef(uint32_t tableId, int64_t pk, uint64_t off, uint32_t bodySize);
    void applyDelete(uint32_t tableId, int64_t pk);
    void addSchema(const TableSchema& schema);
    bool readRowAt(uint64_t off, Row& out);
    bool decodeRowAt(uint32_t tableId, uint64_t off, Row& out);

    // transaction
    void ensureImplicitTxn();
    void commitTxn();
    void discardTxn();
    QueryResult autoCommit(const std::function<QueryResult()>& fn);
    TxnTable& txnTable(uint32_t id);
    bool txnLookup(uint32_t tableId, int64_t pk, Row& out, bool& deleted);
    void stageDelete(uint32_t tableId, int64_t pk);

    // schema / rows
    TableSchema* schemaByName(const std::string& name);
    TableData* tableById(uint32_t id);
    bool pendingTable(uint32_t id) const;
    bool lookupRow(uint32_t tableId, int64_t pk, Row& out);
    void scanRows(uint32_t tableId, const std::function<bool(const Row&)>& fn);
    void maybeCache(TableData& table);
    void dropCache(TableData& table);

    // columnar TopK scan cache
    bool ensureScanIndex(TableData& t, int vecCol);
    void ensureFilterCodes(TableData& t, int col, bool isInt);
    bool filterCode(const TableData& t, int type, int64_t i, const std::string& s,
                    int32_t& code) const;
    int32_t codeForValue(TableData& t, const Value& v);
    void touchScanRow(TableData& t, size_t slot, const Row& row);
    void moveScanRow(TableData& t, size_t from, size_t to);
    void popScanRow(TableData& t);
    void resetScanIndex(TableData& t);

    // statements
    QueryResult execCreateTable(const CreateTableStmt& st);
    QueryResult execCreateIndex(const CreateIndexStmt& st);
    QueryResult execInsert(const InsertStmt& st);
    QueryResult execSelect(const SelectStmt& st, const std::string& searchMode);
    QueryResult execUpdate(const UpdateStmt& st);
    QueryResult execDelete(const DeleteStmt& st);
    QueryResult execBegin();
    QueryResult execCommit();
    QueryResult execRollback();

    std::string dir_;
    uint64_t cacheBudget_;
    uint64_t checkpointBytes_ = 64ull * 1024 * 1024;
    std::string activeFile_;
    uint64_t epoch_ = 0;
    FILE* wal_ = nullptr;
    FILE* readFp_ = nullptr;
    uint64_t walSize_ = 0;
    uint64_t nextTxnId_ = 1;
    uint64_t nextTableId_ = 1;
    std::unordered_map<std::string, uint32_t> tableByName_;
    std::unordered_map<uint32_t, std::unique_ptr<TableData>> tables_;
    uint64_t cachedTotal_ = 0;
    bool cacheDisabled_ = false;
    bool implicit_ = false;
    Txn txn_;
};

// encoding helpers shared by engine/storage
uint32_t crc32(const uint8_t* data, size_t len);
uint64_t rowBodySize(const TableSchema& schema, const Row& row);
void encodeRowBody(const TableSchema& schema, const Row& row, std::vector<uint8_t>& out);
Row decodeRowBody(const TableSchema& schema, const uint8_t*& p, const uint8_t* end);
void encodeSchema(const TableSchema& schema, std::vector<uint8_t>& out);
TableSchema decodeSchema(const uint8_t*& p, const uint8_t* end);
std::string joinPath(const std::string& dir, const std::string& name);

}  // namespace mvdb
