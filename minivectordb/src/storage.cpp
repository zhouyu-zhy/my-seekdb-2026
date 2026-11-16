#include "mvdb.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>

#ifdef _WIN32
#include <io.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace mvdb {

namespace fs = std::filesystem;

namespace {

uint32_t crcTable[256];
bool crcReady = false;

void initCrc() {
    if (crcReady) return;
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t c = i;
        for (int k = 0; k < 8; ++k) c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
        crcTable[i] = c;
    }
    crcReady = true;
}

void putU32(std::vector<uint8_t>& out, uint32_t v) {
    out.push_back(static_cast<uint8_t>(v));
    out.push_back(static_cast<uint8_t>(v >> 8));
    out.push_back(static_cast<uint8_t>(v >> 16));
    out.push_back(static_cast<uint8_t>(v >> 24));
}

void putU64(std::vector<uint8_t>& out, uint64_t v) {
    for (int i = 0; i < 8; ++i) out.push_back(static_cast<uint8_t>(v >> (8 * i)));
}

void putI64(std::vector<uint8_t>& out, int64_t v) { putU64(out, static_cast<uint64_t>(v)); }

void putStr(std::vector<uint8_t>& out, const std::string& s) {
    putU32(out, static_cast<uint32_t>(s.size()));
    out.insert(out.end(), s.begin(), s.end());
}

uint32_t getU32(const uint8_t*& p) {
    uint32_t v = static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
                 (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
    p += 4;
    return v;
}

uint64_t getU64(const uint8_t*& p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(p[i]) << (8 * i);
    p += 8;
    return v;
}

int64_t getI64(const uint8_t*& p) { return static_cast<int64_t>(getU64(p)); }

std::string getStr(const uint8_t*& p, const uint8_t* end) {
    if (end - p < 4) fail("internal_error", "corrupt string length");
    uint32_t len = getU32(p);
    if (static_cast<uint64_t>(end - p) < len) fail("internal_error", "corrupt string payload");
    std::string s(reinterpret_cast<const char*>(p), len);
    p += len;
    return s;
}

void putFloat(std::vector<uint8_t>& out, float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    putU32(out, bits);
}

float getFloat(const uint8_t*& p) {
    uint32_t bits = getU32(p);
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

bool writeAll(FILE* f, const void* data, size_t n) {
    return n == 0 || std::fwrite(data, 1, n, f) == n;
}

void syncFile(FILE* f) {
    if (std::fflush(f) != 0) fail("io_error", "flush failed");
#ifdef _WIN32
    if (_commit(_fileno(f)) != 0) fail("io_error", "fsync failed");
#elif defined(__linux__)
    if (::fdatasync(::fileno(f)) != 0) fail("io_error", "fdatasync failed");
#else
    if (::fsync(::fileno(f)) != 0) fail("io_error", "fsync failed");
#endif
}

bool seekFile(FILE* f, uint64_t off) {
#ifdef _WIN32
    return _fseeki64(f, static_cast<__int64>(off), SEEK_SET) == 0;
#else
    return ::fseeko(f, static_cast<off_t>(off), SEEK_SET) == 0;
#endif
}

uint64_t fileSize(FILE* f) {
#ifdef _WIN32
    __int64 c = _ftelli64(f);
    if (c < 0) return 0;
    if (_fseeki64(f, 0, SEEK_END) != 0) return 0;
    __int64 e = _ftelli64(f);
    _fseeki64(f, c, SEEK_SET);
    return e < 0 ? 0 : static_cast<uint64_t>(e);
#else
    off_t c = ::ftello(f);
    if (c < 0) return 0;
    if (::fseeko(f, 0, SEEK_END) != 0) return 0;
    off_t e = ::ftello(f);
    ::fseeko(f, static_cast<off_t>(c), SEEK_SET);
    return e < 0 ? 0 : static_cast<uint64_t>(e);
#endif
}

bool readAt(FILE* f, uint64_t off, void* buf, size_t n) {
    if (!seekFile(f, off)) return false;
    return n == 0 || std::fread(buf, 1, n, f) == n;
}

void truncateFile(FILE* f, uint64_t size) {
    if (std::fflush(f) != 0) fail("io_error", "flush failed");
#ifdef _WIN32
    if (_chsize_s(_fileno(f), static_cast<__int64>(size)) != 0) fail("io_error", "truncate failed");
    if (_commit(_fileno(f)) != 0) fail("io_error", "fsync failed");
#else
    if (::ftruncate(::fileno(f), static_cast<off_t>(size)) != 0) fail("io_error", "truncate failed");
    if (::fsync(::fileno(f)) != 0) fail("io_error", "fsync failed");
#endif
}

std::string lowerKey(const std::string& s) {
    std::string out = s;
    for (char& c : out) {
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    }
    return out;
}

uint64_t decodedRowBytes(const TableSchema& schema, const Row& row) {
    uint64_t n = 48 + 48ull * schema.cols.size();
    for (const Value& v : row) {
        n += 24 + v.s.size() + v.v.size() * sizeof(float);
    }
    return n;
}

struct RawBlockWriter {
    FILE* f = nullptr;
    uint64_t pos = 0;

    void write(uint64_t txnId, const std::vector<uint8_t>& payload, uint64_t& payloadStart) {
        std::vector<uint8_t> header;
        header.reserve(kBlockHeaderSize);
        putU32(header, kBlockMagic);
        putU64(header, txnId);
        putU64(header, static_cast<uint64_t>(payload.size()));
        putU32(header, crc32(payload.data(), payload.size()));
        putU32(header, 0);
        std::vector<uint8_t> tail;
        tail.reserve(kBlockTailSize);
        putU32(tail, kTailMagic);
        putU64(tail, txnId);
        putU64(tail, static_cast<uint64_t>(payload.size()));
        putU32(tail, crc32(payload.data(), payload.size()));

        payloadStart = pos + kBlockHeaderSize;
        if (!writeAll(f, header.data(), header.size()) ||
            !writeAll(f, payload.data(), payload.size()) ||
            !writeAll(f, tail.data(), tail.size())) {
            fail("io_error", "write failed");
        }
        pos += header.size() + payload.size() + tail.size();
    }
};

}  // namespace

int TableSchema::columnIndex(const std::string& columnName) const {
    for (size_t i = 0; i < cols.size(); ++i) {
        const std::string& a = cols[i].name;
        if (a.size() != columnName.size()) continue;
        bool same = true;
        for (size_t k = 0; k < a.size(); ++k) {
            char ca = a[k], cb = columnName[k];
            if (ca >= 'A' && ca <= 'Z') ca = static_cast<char>(ca - 'A' + 'a');
            if (cb >= 'A' && cb <= 'Z') cb = static_cast<char>(cb - 'A' + 'a');
            if (ca != cb) { same = false; break; }
        }
        if (same) return static_cast<int>(i);
    }
    return -1;
}

int TableSchema::vectorColumnCount() const {
    int n = 0;
    for (const Column& c : cols) if (c.type == ColType::VECTOR) ++n;
    return n;
}

uint32_t crc32(const uint8_t* data, size_t len) {
    initCrc();
    uint32_t c = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; ++i) c = crcTable[(c ^ data[i]) & 0xFFu] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

uint64_t rowBodySize(const TableSchema& schema, const Row& row) {
    uint64_t n = 8;  // duplicated primary key for fast skipping
    for (size_t i = 0; i < schema.cols.size(); ++i) {
        const Column& c = schema.cols[i];
        (void)row;
        if (c.type == ColType::BIGINT) n += 8;
        else if (c.type == ColType::VARCHAR) n += 4 + row[i].s.size();
        else n += 4 + row[i].v.size() * sizeof(float);
    }
    return n;
}

void encodeRowBody(const TableSchema& schema, const Row& row, std::vector<uint8_t>& out) {
    putI64(out, row[schema.pk].i);
    for (size_t i = 0; i < schema.cols.size(); ++i) {
        const Column& c = schema.cols[i];
        const Value& v = row[i];
        if (c.type == ColType::BIGINT) {
            putI64(out, v.i);
        } else if (c.type == ColType::VARCHAR) {
            putStr(out, v.s);
        } else {
            putU32(out, static_cast<uint32_t>(v.v.size()));
            for (float f : v.v) putFloat(out, f);
        }
    }
}

Row decodeRowBody(const TableSchema& schema, const uint8_t*& p, const uint8_t* end) {
    if (end - p < 8) fail("internal_error", "corrupt row");
    int64_t pk = getI64(p);
    Row row(schema.cols.size());
    if (schema.pk >= 0) row[static_cast<size_t>(schema.pk)].i = pk;
    for (size_t i = 0; i < schema.cols.size(); ++i) {
        const Column& c = schema.cols[i];
        if (c.type == ColType::BIGINT) {
            if (end - p < 8) fail("internal_error", "corrupt bigint");
            row[i].i = getI64(p);
        } else if (c.type == ColType::VARCHAR) {
            row[i].s = getStr(p, end);
        } else {
            if (end - p < 4) fail("internal_error", "corrupt vector");
            uint32_t dim = getU32(p);
            if (static_cast<uint64_t>(end - p) < static_cast<uint64_t>(dim) * 4) {
                fail("internal_error", "corrupt vector data");
            }
            row[i].v.resize(dim);
            for (uint32_t k = 0; k < dim; ++k) row[i].v[k] = getFloat(p);
        }
    }
    return row;
}

void encodeSchema(const TableSchema& schema, std::vector<uint8_t>& out) {
    putU32(out, schema.id);
    putStr(out, schema.name);
    putU32(out, static_cast<uint32_t>(schema.cols.size()));
    for (const Column& c : schema.cols) {
        putStr(out, c.name);
        out.push_back(static_cast<uint8_t>(c.type));
        putU32(out, c.length);
    }
    putU32(out, static_cast<uint32_t>(static_cast<int32_t>(schema.pk)));
    putU32(out, static_cast<uint32_t>(static_cast<int32_t>(schema.vec)));
    putU32(out, static_cast<uint32_t>(schema.indexes.size()));
    for (const VectorIndexDef& idx : schema.indexes) {
        putStr(out, idx.name);
        putStr(out, idx.column);
    }
}

TableSchema decodeSchema(const uint8_t*& p, const uint8_t* end) {
    TableSchema schema;
    if (end - p < 4) fail("internal_error", "corrupt schema");
    schema.id = getU32(p);
    schema.name = getStr(p, end);
    if (end - p < 4) fail("internal_error", "corrupt schema");
    uint32_t ncols = getU32(p);
    if (ncols > 65535) fail("internal_error", "corrupt schema");
    for (uint32_t i = 0; i < ncols; ++i) {
        Column c;
        c.name = getStr(p, end);
        if (p >= end) fail("internal_error", "corrupt schema");
        c.type = static_cast<ColType>(*p++);
        if (end - p < 4) fail("internal_error", "corrupt schema");
        c.length = getU32(p);
        schema.cols.push_back(std::move(c));
    }
    if (end - p < 4) fail("internal_error", "corrupt schema");
    schema.pk = static_cast<int32_t>(getU32(p));
    if (end - p < 4) fail("internal_error", "corrupt schema");
    schema.vec = static_cast<int32_t>(getU32(p));
    if (end - p < 4) fail("internal_error", "corrupt schema");
    uint32_t nidx = getU32(p);
    if (nidx > 65535) fail("internal_error", "corrupt schema");
    for (uint32_t i = 0; i < nidx; ++i) {
        VectorIndexDef def;
        def.name = getStr(p, end);
        def.column = getStr(p, end);
        schema.indexes.push_back(std::move(def));
    }
    return schema;
}

// ---------------------------------------------------------------------------
// Engine storage
// ---------------------------------------------------------------------------

Engine::Engine(std::string dataDir, uint64_t cacheBudget)
    : dir_(std::move(dataDir)), cacheBudget_(cacheBudget) {
    while (!dir_.empty() && (dir_.back() == '/' || dir_.back() == '\\')) dir_.pop_back();
    if (dir_.empty()) dir_ = ".";
    if (const char* env = std::getenv("MVDB_CHECKPOINT_MB")) {
        try {
            uint64_t mb = static_cast<uint64_t>(std::stoull(env));
            if (mb > 0) checkpointBytes_ = mb * 1024 * 1024;
        } catch (...) {
        }
    }
}

Engine::~Engine() { closeFiles(); }

std::string joinPath(const std::string& dir, const std::string& name) { return dir + "/" + name; }

void Engine::closeFiles() {
    if (wal_) {
        std::fclose(wal_);
        wal_ = nullptr;
    }
    if (readFp_) {
        std::fclose(readFp_);
        readFp_ = nullptr;
    }
}

void Engine::initStorage() {
    std::error_code ec;
    fs::create_directories(dir_, ec);
    if (!fs::exists(dir_)) fail("io_error", "cannot create data directory");

    const std::string catalogPath = joinPath(dir_, "catalog.dat");
    bool haveCatalog = false;
    if (fs::exists(catalogPath)) {
        std::ifstream in(catalogPath, std::ios::binary);
        std::string magic, epochLine, fileLine;
        if (std::getline(in, magic) && magic == "MVDB-CATALOG-1" && std::getline(in, epochLine) &&
            std::getline(in, fileLine)) {
            if (epochLine.rfind("epoch ", 0) == 0 && fileLine.rfind("file ", 0) == 0) {
                try {
                    epoch_ = static_cast<uint64_t>(std::stoull(epochLine.substr(6)));
                    activeFile_ = fileLine.substr(5);
                    haveCatalog = !activeFile_.empty();
                } catch (...) {
                    haveCatalog = false;
                }
            }
        }
    }
    if (!haveCatalog) {
        epoch_ = 0;
        activeFile_ = "rows_0.dat";
        std::string path = joinPath(dir_, activeFile_);
        FILE* f = std::fopen(path.c_str(), "w+b");
        if (!f) fail("io_error", "cannot create data file");
        std::fclose(f);
        writeCatalogAtomic();
    }

    if (!fs::exists(joinPath(dir_, activeFile_))) {
        fail("io_error", "active data file is missing");
    }
    cleanupOrphanFiles();
    openActiveFile();
}

void Engine::cleanupOrphanFiles() {
    std::error_code ec;
    for (auto it = fs::directory_iterator(dir_, ec); !ec && it != fs::directory_iterator(); it.increment(ec)) {
        if (!it->is_regular_file()) continue;
        std::string name = it->path().filename().string();
        if (name.size() > 4 && name.compare(name.size() - 4, 4, ".tmp") == 0) {
            fs::remove(it->path(), ec);
            continue;
        }
        if (name.rfind("rows_", 0) == 0 && name.size() > 4 && name.compare(name.size() - 4, 4, ".dat") == 0 &&
            name != activeFile_) {
            fs::remove(it->path(), ec);
        }
    }
}

void Engine::openActiveFile() {
    std::string path = joinPath(dir_, activeFile_);
    wal_ = std::fopen(path.c_str(), "r+b");
    if (!wal_) {
        wal_ = std::fopen(path.c_str(), "w+b");
        if (!wal_) fail("io_error", "cannot open data file");
    }
    walSize_ = fileSize(wal_);
    seekFile(wal_, walSize_);
    reopenReadFile();
}

void Engine::reopenReadFile() {
    if (readFp_) {
        std::fclose(readFp_);
        readFp_ = nullptr;
    }
    std::string path = joinPath(dir_, activeFile_);
    readFp_ = std::fopen(path.c_str(), "rb");
    if (!readFp_) fail("io_error", "cannot open data file for reading");
}

void Engine::writeCatalogAtomic() {
    const std::string tmp = joinPath(dir_, "catalog.tmp");
    const std::string target = joinPath(dir_, "catalog.dat");
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) fail("io_error", "cannot write catalog");
    std::string text = "MVDB-CATALOG-1\nepoch " + std::to_string(epoch_) + "\nfile " + activeFile_ + "\n";
    if (!writeAll(f, text.data(), text.size())) {
        std::fclose(f);
        fail("io_error", "catalog write failed");
    }
    syncFile(f);
    std::fclose(f);
#ifdef _WIN32
    std::error_code ec;
    fs::remove(target, ec);
    fs::rename(tmp, target, ec);
    if (ec) fail("io_error", "catalog rename failed");
#else
    if (::rename(tmp.c_str(), target.c_str()) != 0) fail("io_error", "catalog rename failed");
    int dfd = ::open(dir_.c_str(), O_RDONLY | O_DIRECTORY);
    if (dfd >= 0) {
        ::fsync(dfd);
        ::close(dfd);
    }
#endif
}

void Engine::truncateWal(uint64_t size) {
    if (size >= walSize_) return;
    truncateFile(wal_, size);
    walSize_ = size;
}

void Engine::recover() {
    initStorage();
    recoverStorage();
    if (walSize_ > checkpointBytes_ * 2 && walSize_ > 4 * liveBytesTotal() + 1) {
        checkpoint();
    }
}

void Engine::recoverStorage() {
    uint64_t pos = 0;
    std::vector<uint8_t> payload;
    while (pos < walSize_) {
        uint8_t header[kBlockHeaderSize];
        if (!readAt(readFp_, pos, header, sizeof(header))) {
            truncateWal(pos);
            break;
        }
        const uint8_t* hp = header;
        uint32_t magic = getU32(hp);
        uint64_t txnId = getU64(hp);
        uint64_t payloadLen = getU64(hp);
        uint32_t crc = getU32(hp);
        if (magic != kBlockMagic || payloadLen > (1ull << 34) ||
            pos + kBlockHeaderSize + payloadLen + kBlockTailSize > walSize_) {
            truncateWal(pos);
            break;
        }
        payload.resize(static_cast<size_t>(payloadLen));
        if (payloadLen && !readAt(readFp_, pos + kBlockHeaderSize, payload.data(), payload.size())) {
            truncateWal(pos);
            break;
        }
        uint8_t tail[kBlockTailSize];
        if (!readAt(readFp_, pos + kBlockHeaderSize + payloadLen, tail, sizeof(tail))) {
            truncateWal(pos);
            break;
        }
        const uint8_t* tp = tail;
        uint32_t tmagic = getU32(tp);
        uint64_t ttxn = getU64(tp);
        uint64_t tlen = getU64(tp);
        uint32_t tcrc = getU32(tp);
        if (tmagic != kTailMagic || ttxn != txnId || tlen != payloadLen ||
            tcrc != crc || crc != crc32(payload.data(), payload.size())) {
            truncateWal(pos);
            break;
        }
        if (txnId != 0) nextTxnId_ = std::max(nextTxnId_, txnId + 1);
        applyPayload(payload, pos + kBlockHeaderSize);
        pos += kBlockHeaderSize + payloadLen + kBlockTailSize;
    }
    if (pos < walSize_) truncateWal(pos);
    seekFile(wal_, walSize_);
}

void Engine::appendBlock(const std::vector<uint8_t>& payload, uint64_t* blockStart, uint64_t* payloadStart) {
    if (blockStart) *blockStart = walSize_;
    if (payloadStart) *payloadStart = walSize_ + kBlockHeaderSize;
    if (payload.empty()) return;
    uint64_t txnId = nextTxnId_++;
    std::vector<uint8_t> header;
    header.reserve(kBlockHeaderSize);
    putU32(header, kBlockMagic);
    putU64(header, txnId);
    putU64(header, static_cast<uint64_t>(payload.size()));
    uint32_t sum = crc32(payload.data(), payload.size());
    putU32(header, sum);
    putU32(header, 0);
    std::vector<uint8_t> tail;
    tail.reserve(kBlockTailSize);
    putU32(tail, kTailMagic);
    putU64(tail, txnId);
    putU64(tail, static_cast<uint64_t>(payload.size()));
    putU32(tail, sum);

    if (!seekFile(wal_, walSize_)) fail("io_error", "seek failed");
    if (!writeAll(wal_, header.data(), header.size()) || !writeAll(wal_, payload.data(), payload.size()) ||
        !writeAll(wal_, tail.data(), tail.size())) {
        fail("io_error", "write failed");
    }
    syncFile(wal_);
    walSize_ += header.size() + payload.size() + tail.size();
}

uint64_t Engine::liveBytesTotal() const {
    uint64_t n = 0;
    for (const auto& kv : tables_) n += kv.second->liveBytes;
    return n;
}

void Engine::addSchema(const TableSchema& schema) {
    auto it = tables_.find(schema.id);
    if (it != tables_.end()) {
        it->second->schema = schema;
        return;
    }
    auto td = std::make_unique<TableData>();
    td->schema = schema;
    td->cached = !cacheDisabled_ && cacheBudget_ > 0 && walSize_ <= cacheBudget_ / 2;
    tableByName_[lowerKey(schema.name)] = schema.id;
    nextTableId_ = std::max<uint64_t>(nextTableId_, static_cast<uint64_t>(schema.id) + 1);
    tables_[schema.id] = std::move(td);
}

void Engine::buildPayload(std::vector<uint8_t>& payload) {
    payload.clear();
    auto appendFrame = [&](uint8_t type, const std::function<void()>& body) {
        size_t start = payload.size();
        payload.push_back(type);
        putU32(payload, 0);
        body();
        uint32_t len = static_cast<uint32_t>(payload.size() - start);
        payload[start + 1] = static_cast<uint8_t>(len);
        payload[start + 2] = static_cast<uint8_t>(len >> 8);
        payload[start + 3] = static_cast<uint8_t>(len >> 16);
        payload[start + 4] = static_cast<uint8_t>(len >> 24);
    };

    auto appendTableRows = [&](uint32_t tableId) {
        auto ot = txn_.tables.find(tableId);
        if (ot == txn_.tables.end()) return;
        std::vector<int64_t> pks;
        pks.reserve(ot->second.entries.size());
        for (const auto& kv : ot->second.entries) pks.push_back(kv.first);
        std::sort(pks.begin(), pks.end());
        for (int64_t pk : pks) {
            const OverlayEntry& e = ot->second.entries.at(pk);
            if (e.deleted) {
                appendFrame(OP_DELETE, [&]() {
                    putU32(payload, tableId);
                    putI64(payload, pk);
                });
            } else {
                const TableSchema* schema = nullptr;
                auto existing = tables_.find(tableId);
                if (existing != tables_.end()) {
                    schema = &existing->second->schema;
                } else {
                    for (const TableSchema& s : txn_.newTables) {
                        if (s.id == tableId) { schema = &s; break; }
                    }
                }
                if (!schema) fail("internal_error", "row for unknown table");
                appendFrame(OP_UPSERT, [&]() {
                    putU32(payload, tableId);
                    encodeRowBody(*schema, e.row, payload);
                });
            }
        }
    };

    for (const TableSchema& schema : txn_.newTables) {
        appendFrame(OP_CREATE_TABLE, [&]() { encodeSchema(schema, payload); });
        appendTableRows(schema.id);
    }

    std::vector<uint32_t> existing;
    for (const auto& kv : txn_.tables) {
        bool isNew = false;
        for (const TableSchema& s : txn_.newTables) {
            if (s.id == kv.first) { isNew = true; break; }
        }
        if (!isNew) existing.push_back(kv.first);
    }
    std::sort(existing.begin(), existing.end());
    for (uint32_t id : existing) appendTableRows(id);

    for (const auto& kv : txn_.newIndexes) {
        const VectorIndexDef& def = kv.second;
        appendFrame(OP_CREATE_INDEX, [&]() {
            putU32(payload, kv.first);
            putStr(payload, def.name);
            putStr(payload, def.column);
        });
    }
}

void Engine::applyPayload(const std::vector<uint8_t>& payload, uint64_t payloadOffset) {
    const uint8_t* p = payload.data();
    const uint8_t* end = p + payload.size();
    while (p < end) {
        size_t frameStart = static_cast<size_t>(p - payload.data());
        if (end - p < 5) fail("internal_error", "corrupt operation");
        uint8_t type = *p++;
        uint32_t len = getU32(p);
        if (len < 5 || static_cast<uint64_t>(len) > static_cast<uint64_t>(end - payload.data() - frameStart)) {
            fail("internal_error", "corrupt operation length");
        }
        const uint8_t* frameEnd = payload.data() + frameStart + len;
        if (type == OP_CREATE_TABLE) {
            TableSchema schema = decodeSchema(p, frameEnd);
            addSchema(schema);
        } else if (type == OP_UPSERT) {
            if (frameEnd - p < 4) fail("internal_error", "corrupt upsert");
            uint32_t tableId = getU32(p);
            auto it = tables_.find(tableId);
            if (it == tables_.end()) fail("internal_error", "row for unknown table");
            TableData* t = it->second.get();
            const uint8_t* rowStart = p;
            if (t->cached) {
                Row row = decodeRowBody(t->schema, p, frameEnd);
                uint32_t bodySize = static_cast<uint32_t>(frameEnd - rowStart);
                applyUpsertRow(tableId, row, payloadOffset + frameStart, bodySize);
            } else {
                if (frameEnd - p < 8) fail("internal_error", "corrupt row key");
                const uint8_t* q = rowStart;
                int64_t pk = getI64(q);
                uint32_t bodySize = static_cast<uint32_t>(frameEnd - rowStart);
                applyUpsertRef(tableId, pk, payloadOffset + frameStart, bodySize);
            }
        } else if (type == OP_DELETE) {
            if (frameEnd - p < 12) fail("internal_error", "corrupt delete");
            uint32_t tableId = getU32(p);
            int64_t pk = getI64(p);
            applyDelete(tableId, pk);
        } else if (type == OP_CREATE_INDEX) {
            if (frameEnd - p < 4) fail("internal_error", "corrupt index");
            uint32_t tableId = getU32(p);
            VectorIndexDef def;
            def.name = getStr(p, frameEnd);
            def.column = getStr(p, frameEnd);
            auto it = tables_.find(tableId);
            if (it != tables_.end()) {
                bool exists = false;
                for (const VectorIndexDef& old : it->second->schema.indexes) {
                    if (lowerKey(old.name) == lowerKey(def.name)) { exists = true; break; }
                }
                if (!exists) it->second->schema.indexes.push_back(def);
            }
        } else {
            fail("internal_error", "unknown operation");
        }
        p = frameEnd;
    }
}

void Engine::applyUpsertRow(uint32_t tableId, Row& row, uint64_t off, uint32_t bodySize) {
    auto it = tables_.find(tableId);
    if (it == tables_.end()) fail("internal_error", "row for unknown table");
    TableData& t = *it->second;
    t.vecCacheValid = false;
    int64_t pk = row[static_cast<size_t>(t.schema.pk)].i;
    auto r = t.refs.find(pk);
    uint32_t oldSize = r == t.refs.end() ? 0 : r->second.size;
    if (t.cached) {
        auto s = t.slotOf.find(pk);
        if (s == t.slotOf.end()) {
            size_t slot = t.rows.size();
            t.rows.push_back(std::move(row));
            t.slotOf[pk] = static_cast<int>(slot);
            uint64_t add = decodedRowBytes(t.schema, t.rows.back());
            t.cachedBytes += add;
            cachedTotal_ += add;
        } else {
            Row& old = t.rows[static_cast<size_t>(s->second)];
            uint64_t before = decodedRowBytes(t.schema, old);
            uint64_t after = decodedRowBytes(t.schema, row);
            old = std::move(row);
            t.cachedBytes = t.cachedBytes + after - std::min<uint64_t>(t.cachedBytes, before);
            if (after >= before) cachedTotal_ += after - before;
            else cachedTotal_ -= std::min<uint64_t>(cachedTotal_, before - after);
        }
    }
    t.refs[pk] = DiskRef{off, bodySize};
    t.liveBytes = t.liveBytes + bodySize - std::min<uint64_t>(t.liveBytes, oldSize);

    if (t.cached && cacheBudget_ > 0 && cachedTotal_ > cacheBudget_ * 3 / 4) {
        for (auto& kv : tables_) dropCache(*kv.second);
        cacheDisabled_ = true;
    }
}

void Engine::applyUpsertRef(uint32_t tableId, int64_t pk, uint64_t off, uint32_t bodySize) {
    auto it = tables_.find(tableId);
    if (it == tables_.end()) fail("internal_error", "row for unknown table");
    TableData& t = *it->second;
    auto r = t.refs.find(pk);
    uint32_t oldSize = r == t.refs.end() ? 0 : r->second.size;
    t.refs[pk] = DiskRef{off, bodySize};
    t.liveBytes = t.liveBytes + bodySize - std::min<uint64_t>(t.liveBytes, oldSize);
}

void Engine::applyDelete(uint32_t tableId, int64_t pk) {
    auto it = tables_.find(tableId);
    if (it == tables_.end()) return;
    TableData& t = *it->second;
    t.vecCacheValid = false;
    auto r = t.refs.find(pk);
    if (r == t.refs.end()) return;
    uint32_t oldSize = r->second.size;
    t.liveBytes -= std::min<uint64_t>(t.liveBytes, oldSize);
    t.refs.erase(r);
    if (t.cached) {
        auto s = t.slotOf.find(pk);
        if (s != t.slotOf.end()) {
            size_t slot = static_cast<size_t>(s->second);
            uint64_t bytes = decodedRowBytes(t.schema, t.rows[slot]);
            t.cachedBytes -= std::min<uint64_t>(t.cachedBytes, bytes);
            cachedTotal_ -= std::min<uint64_t>(cachedTotal_, bytes);
            size_t last = t.rows.size() - 1;
            if (slot != last) {
                t.rows[slot] = std::move(t.rows[last]);
                int64_t movedPk = t.rows[slot][static_cast<size_t>(t.schema.pk)].i;
                t.slotOf[movedPk] = static_cast<int>(slot);
            }
            t.rows.pop_back();
            t.slotOf.erase(pk);
        }
    }
}

void Engine::dropCache(TableData& t) {
    if (!t.cached) return;
    cachedTotal_ -= std::min<uint64_t>(cachedTotal_, t.cachedBytes);
    t.cachedBytes = 0;
    t.rows.clear();
    t.rows.shrink_to_fit();
    t.slotOf.clear();
    t.vecCache.clear();
    t.vecCache.shrink_to_fit();
    t.vecRows.clear();
    t.vecRows.shrink_to_fit();
    t.vecCacheValid = false;
    t.cached = false;
}

void Engine::maybeCache(TableData& t) {
    if (t.cached || cacheDisabled_ || cacheBudget_ == 0) return;
    if (t.liveBytes * 2 + 1024 <= cacheBudget_) {
        // Kept for completeness; normal caching happens while replaying rows.
        t.cached = true;
    }
}

bool Engine::readRowAt(uint64_t off, Row& out) {
    uint8_t fh[5];
    if (!readAt(readFp_, off, fh, sizeof(fh))) return false;
    const uint8_t* fp = fh;
    uint8_t type = *fp++;
    uint32_t len = getU32(fp);
    if (type != OP_UPSERT || len < 5) return false;
    std::vector<uint8_t> body(len - 5);
    if (!readAt(readFp_, off + 5, body.data(), body.size())) return false;
    const uint8_t* p = body.data();
    const uint8_t* end = p + body.size();
    if (end - p < 4) return false;
    uint32_t tableId = getU32(p);
    auto it = tables_.find(tableId);
    if (it == tables_.end()) return false;
    out = decodeRowBody(it->second->schema, p, end);
    return true;
}

bool Engine::decodeRowAt(uint32_t tableId, uint64_t off, Row& out) {
    auto it = tables_.find(tableId);
    if (it == tables_.end()) return false;
    uint8_t fh[5];
    if (!readAt(readFp_, off, fh, sizeof(fh))) return false;
    const uint8_t* fp = fh;
    uint8_t type = *fp++;
    uint32_t len = getU32(fp);
    if (type != OP_UPSERT || len < 5) return false;
    std::vector<uint8_t> body(len - 5);
    if (!readAt(readFp_, off + 5, body.data(), body.size())) return false;
    const uint8_t* p = body.data();
    const uint8_t* end = p + body.size();
    if (end - p < 4) return false;
    getU32(p);  // table id
    out = decodeRowBody(it->second->schema, p, end);
    return true;
}

void Engine::checkpoint() {
    if (txn_.active) return;
    std::error_code ec;
    const std::string newName = "rows_" + std::to_string(epoch_ + 1) + ".dat";
    const std::string newPath = joinPath(dir_, newName);
    FILE* f = std::fopen(newPath.c_str(), "w+b");
    if (!f) fail("io_error", "cannot create checkpoint file");

    RawBlockWriter writer{f, 0};
    auto appendFrame = [](std::vector<uint8_t>& payload, uint8_t type, const std::function<void()>& body) {
        size_t start = payload.size();
        payload.push_back(type);
        putU32(payload, 0);
        body();
        uint32_t len = static_cast<uint32_t>(payload.size() - start);
        payload[start + 1] = static_cast<uint8_t>(len);
        payload[start + 2] = static_cast<uint8_t>(len >> 8);
        payload[start + 3] = static_cast<uint8_t>(len >> 16);
        payload[start + 4] = static_cast<uint8_t>(len >> 24);
    };

    for (auto& kv : tables_) {
        TableData& t = *kv.second;
        std::unordered_map<int64_t, DiskRef> newRefs;
        newRefs.reserve(t.refs.size());

        std::vector<uint8_t> payload;
        appendFrame(payload, OP_CREATE_TABLE, [&]() { encodeSchema(t.schema, payload); });
        uint64_t payloadStart = 0;
        writer.write(0, payload, payloadStart);
        payload.clear();

        std::vector<std::pair<int64_t, size_t>> offsets;
        offsets.reserve(512);
        auto flushRows = [&]() {
            if (payload.empty()) return;
            uint64_t ps = 0;
            writer.write(0, payload, ps);
            for (const auto& o : offsets) {
                newRefs[o.first] = DiskRef{ps + o.second, 0};
            }
            payload.clear();
            offsets.clear();
        };

        scanRows(t.schema.id, [&](const Row& row) {
            int64_t pk = row[static_cast<size_t>(t.schema.pk)].i;
            size_t frameStart = payload.size();
            appendFrame(payload, OP_UPSERT, [&]() {
                putU32(payload, t.schema.id);
                encodeRowBody(t.schema, row, payload);
            });
            offsets.emplace_back(pk, frameStart);
            if (payload.size() >= 4 * 1024 * 1024) flushRows();
            return true;
        });
        flushRows();

        // The checkpoint's CREATE TABLE frame already carries the index list.
        if (!payload.empty()) {
            uint64_t ps = 0;
            writer.write(0, payload, ps);
        }

        // Fill in body sizes from the in-memory rows when they are cached.
        if (t.cached) {
            for (const auto& row : t.rows) {
                int64_t pk = row[static_cast<size_t>(t.schema.pk)].i;
                auto r = newRefs.find(pk);
                if (r != newRefs.end()) r->second.size = static_cast<uint32_t>(rowBodySize(t.schema, row));
            }
        } else {
            for (auto& r : newRefs) {
                auto old = t.refs.find(r.first);
                if (old != t.refs.end()) r.second.size = old->second.size;
            }
        }
        t.refs = std::move(newRefs);
    }

    syncFile(f);
    uint64_t newSize = writer.pos;
    std::fclose(f);

    uint64_t oldEpoch = epoch_;
    std::string oldFile = activeFile_;
    epoch_ = oldEpoch + 1;
    activeFile_ = newName;
    writeCatalogAtomic();

    closeFiles();
    fs::remove(joinPath(dir_, oldFile), ec);
    openActiveFile();
    walSize_ = newSize;
    seekFile(wal_, walSize_);
}

bool Engine::lookupRow(uint32_t tableId, int64_t pk, Row& out) {
    bool deleted = false;
    if (txnLookup(tableId, pk, out, deleted)) return !deleted;
    auto it = tables_.find(tableId);
    if (it == tables_.end()) return false;
    TableData& t = *it->second;
    if (t.cached) {
        auto s = t.slotOf.find(pk);
        if (s == t.slotOf.end()) return false;
        out = t.rows[static_cast<size_t>(s->second)];
        return true;
    }
    auto r = t.refs.find(pk);
    if (r == t.refs.end()) return false;
    return decodeRowAt(tableId, r->second.off, out);
}

}  // namespace mvdb
