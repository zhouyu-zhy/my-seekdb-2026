// JSON-lines front end for the judge.
#include "mvdb.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <io.h>
#endif

namespace {

using namespace mvdb;

struct JValue {
    enum Type { NUL, BOOL, NUM, STR, ARR, OBJ } type = NUL;
    bool b = false;
    bool isInt = false;
    int64_t i = 0;
    double d = 0;
    std::string s;
    std::string raw;
    std::vector<JValue> arr;
    std::vector<std::pair<std::string, JValue>> obj;

    const JValue* get(const std::string& key) const {
        for (const auto& kv : obj) {
            if (kv.first == key) return &kv.second;
        }
        return nullptr;
    }
};

class JsonParser {
public:
    explicit JsonParser(const std::string& text) : s_(text) {}

    JValue parse() {
        skip();
        JValue v = value();
        skip();
        if (i_ != s_.size()) throw std::runtime_error("trailing JSON characters");
        return v;
    }

private:
    const std::string& s_;
    size_t i_ = 0;

    void skip() {
        while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\t' || s_[i_] == '\r' || s_[i_] == '\n')) ++i_;
    }
    char cur() const { return i_ < s_.size() ? s_[i_] : '\0'; }

    void expect(char c) {
        if (cur() != c) throw std::runtime_error("invalid JSON");
        ++i_;
    }

    static void appendUtf8(std::string& out, uint32_t cp) {
        if (cp <= 0x7F) {
            out.push_back(static_cast<char>(cp));
        } else if (cp <= 0x7FF) {
            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp <= 0xFFFF) {
            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }

    uint32_t hex4() {
        uint32_t v = 0;
        for (int k = 0; k < 4; ++k) {
            if (i_ >= s_.size()) throw std::runtime_error("bad unicode escape");
            char c = s_[i_++];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= static_cast<uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f') v |= static_cast<uint32_t>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= static_cast<uint32_t>(c - 'A' + 10);
            else throw std::runtime_error("bad unicode escape");
        }
        return v;
    }

    std::string string() {
        expect('"');
        std::string out;
        while (i_ < s_.size()) {
            const size_t begin = i_;
            while (i_ < s_.size() && s_[i_] != '"' && s_[i_] != '\\') {
                if (static_cast<unsigned char>(s_[i_]) < 0x20) {
                    throw std::runtime_error("unescaped JSON control character");
                }
                ++i_;
            }
            out.append(s_, begin, i_ - begin);
            if (i_ >= s_.size()) break;
            if (s_[i_++] == '"') return out;
            if (i_ >= s_.size()) break;
            char e = s_[i_++];
            switch (e) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u': {
                    uint32_t cp = hex4();
                    if (cp >= 0xD800 && cp <= 0xDBFF && i_ + 1 < s_.size() && s_[i_] == '\\' && s_[i_ + 1] == 'u') {
                        i_ += 2;
                        uint32_t low = hex4();
                        if (low >= 0xDC00 && low <= 0xDFFF) {
                            cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                        }
                    }
                    appendUtf8(out, cp);
                    break;
                }
                default: throw std::runtime_error("bad JSON escape");
            }
        }
        throw std::runtime_error("unterminated JSON string");
    }

    JValue value() {
        skip();
        JValue v;
        char c = cur();
        if (c == '{') {
            v.type = JValue::OBJ;
            ++i_;
            skip();
            if (cur() == '}') { ++i_; return v; }
            for (;;) {
                skip();
                std::string key = string();
                skip();
                expect(':');
                JValue child = value();
                v.obj.emplace_back(std::move(key), std::move(child));
                skip();
                if (cur() == ',') { ++i_; continue; }
                expect('}');
                break;
            }
            return v;
        }
        if (c == '[') {
            v.type = JValue::ARR;
            ++i_;
            skip();
            if (cur() == ']') { ++i_; return v; }
            for (;;) {
                v.arr.push_back(value());
                skip();
                if (cur() == ',') { ++i_; continue; }
                expect(']');
                break;
            }
            return v;
        }
        if (c == '"') {
            v.type = JValue::STR;
            v.s = string();
            return v;
        }
        if (c == 't') {
            if (s_.compare(i_, 4, "true") != 0) throw std::runtime_error("invalid JSON");
            i_ += 4;
            v.type = JValue::BOOL;
            v.b = true;
            return v;
        }
        if (c == 'f') {
            if (s_.compare(i_, 5, "false") != 0) throw std::runtime_error("invalid JSON");
            i_ += 5;
            v.type = JValue::BOOL;
            v.b = false;
            return v;
        }
        if (c == 'n') {
            if (s_.compare(i_, 4, "null") != 0) throw std::runtime_error("invalid JSON");
            i_ += 4;
            v.type = JValue::NUL;
            return v;
        }
        if (c == '-' || (c >= '0' && c <= '9')) {
            size_t start = i_;
            if (cur() == '-') ++i_;
            while (i_ < s_.size() && s_[i_] >= '0' && s_[i_] <= '9') ++i_;
            if (cur() == '.') {
                ++i_;
                while (i_ < s_.size() && s_[i_] >= '0' && s_[i_] <= '9') ++i_;
            }
            if (cur() == 'e' || cur() == 'E') {
                ++i_;
                if (cur() == '+' || cur() == '-') ++i_;
                while (i_ < s_.size() && s_[i_] >= '0' && s_[i_] <= '9') ++i_;
            }
            if (i_ == start) throw std::runtime_error("invalid JSON number");
            v.type = JValue::NUM;
            v.raw = s_.substr(start, i_ - start);
            if (v.raw.find('.') == std::string::npos && v.raw.find('e') == std::string::npos &&
                v.raw.find('E') == std::string::npos) {
                char* end = nullptr;
                errno = 0;
                long long n = std::strtoll(v.raw.c_str(), &end, 10);
                if (errno == 0 && end && *end == '\0') {
                    v.isInt = true;
                    v.i = static_cast<int64_t>(n);
                }
            }
            if (!v.isInt) v.d = std::strtod(v.raw.c_str(), nullptr);
            return v;
        }
        throw std::runtime_error("invalid JSON value");
    }
};

void appendJsonString(std::string& out, const std::string& s) {
    out.push_back('"');
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back(static_cast<char>(c));
                }
                break;
        }
    }
    out.push_back('"');
}

std::string formatDouble(double v) {
    if (!std::isfinite(v)) v = 0.0;
    if (v == 0.0) return "0.0";
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.17g", v);
    std::string s(buf);
    if (s.find('.') == std::string::npos && s.find('e') == std::string::npos && s.find('E') == std::string::npos) {
        s += ".0";
    }
    return s;
}

std::string idJson(const JValue* id) {
    if (!id) return "null";
    if (id->type == JValue::NUM) return id->raw;
    if (id->type == JValue::STR) {
        std::string out;
        appendJsonString(out, id->s);
        return out;
    }
    if (id->type == JValue::BOOL) return id->b ? "true" : "false";
    return "null";
}

void writeError(const std::string& id, const std::string& code, const std::string& message) {
    std::string out = "{\"id\":" + id + ",\"ok\":false,\"error\":{\"code\":";
    appendJsonString(out, code);
    out += ",\"message\":";
    appendJsonString(out, message);
    out += "}}";
    std::cout << out << '\n';
    std::cout.flush();
}

void writeResult(const std::string& id, const QueryResult& r) {
    std::string out = "{\"id\":" + id + ",\"ok\":true";
    if (r.hasRows) {
        out += ",\"columns\":[";
        for (size_t i = 0; i < r.columns.size(); ++i) {
            if (i) out += ",";
            appendJsonString(out, r.columns[i]);
        }
        out += "],\"rows\":[";
        for (size_t i = 0; i < r.rows.size(); ++i) {
            if (i) out += ",";
            out += "[";
            for (size_t j = 0; j < r.rows[i].size(); ++j) {
                if (j) out += ",";
                const OutVal& v = r.rows[i][j];
                switch (v.kind) {
                    case OutVal::INT: out += std::to_string(v.i); break;
                    case OutVal::STR: appendJsonString(out, v.s); break;
                    case OutVal::DOUBLE: out += formatDouble(v.d); break;
                    case OutVal::VEC: {
                        out += "[";
                        for (size_t k = 0; k < v.v.size(); ++k) {
                            if (k) out += ",";
                            out += formatDouble(static_cast<double>(v.v[k]));
                        }
                        out += "]";
                        break;
                    }
                }
            }
            out += "]";
        }
        out += "]";
    }
    if (r.hasAffected) out += ",\"affected_rows\":" + std::to_string(r.affected);
    out += "}";
    std::cout << out << '\n';
    std::cout.flush();
}

std::string trimLine(std::string s) {
    while (!s.empty() && (s.back() == '\r' || s.back() == '\n')) s.pop_back();
    return s;
}

uint64_t cacheBudget() {
    uint64_t budget = 256ull * 1024 * 1024;
    if (const char* env = std::getenv("MVDB_CACHE_MB")) {
        try {
            uint64_t mb = static_cast<uint64_t>(std::stoull(env));
            if (mb > 0) budget = mb * 1024 * 1024;
        } catch (...) {
        }
    } else {
        std::ifstream in("/sys/fs/cgroup/memory.max");
        std::string line;
        if (in && std::getline(in, line) && line != "max") {
            try {
                uint64_t limit = static_cast<uint64_t>(std::stoull(line));
                budget = limit / 4;
                if (budget < 8ull * 1024 * 1024) budget = 8ull * 1024 * 1024;
                if (budget > 256ull * 1024 * 1024) budget = 256ull * 1024 * 1024;
            } catch (...) {
            }
        }
    }
    return budget;
}

}  // namespace

int main(int argc, char** argv) {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    std::string dataDir;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--data-dir" && i + 1 < argc) {
            dataDir = argv[++i];
        } else if (arg.rfind("--data-dir=", 0) == 0) {
            dataDir = arg.substr(11);
        }
    }
    if (dataDir.empty()) {
        if (const char* env = std::getenv("VECTORDB_DATA_DIR")) dataDir = env;
    }
    if (dataDir.empty()) dataDir = "/data";

    try {
        mvdb::Engine engine(dataDir, cacheBudget());
        engine.recover();
        std::cout << "{\"type\":\"ready\"}\n";
        std::cout.flush();

        std::string line;
        while (std::getline(std::cin, line)) {
            line = trimLine(std::move(line));
            if (line.empty()) continue;
            JValue req;
            std::string id = "null";
            try {
                req = JsonParser(line).parse();
                const JValue* idv = req.get("id");
                if (idv) id = idJson(idv);
            } catch (const std::exception& e) {
                writeError("null", "protocol_error", e.what());
                continue;
            }
            const JValue* op = req.get("op");
            std::string opName = (op && op->type == JValue::STR) ? op->s : "execute";
            if (opName == "shutdown") {
                engine.shutdown();
                std::cout << "{\"id\":" << id << ",\"ok\":true}\n";
                std::cout.flush();
                return 0;
            }
            if (opName != "execute") {
                writeError(id, "protocol_error", "unsupported op");
                continue;
            }
            const JValue* sqlv = req.get("sql");
            if (!sqlv || sqlv->type != JValue::STR) {
                writeError(id, "protocol_error", "missing sql");
                continue;
            }
            std::string mode = "exact";
            const JValue* mv = req.get("search_mode");
            if (mv && mv->type == JValue::STR) mode = mv->s;
            try {
                mvdb::Statement st = mvdb::parseStatement(sqlv->s);
                mvdb::QueryResult result = engine.run(st, mode);
                writeResult(id, result);
            } catch (const mvdb::DbError& e) {
                writeError(id, e.code, e.what());
            } catch (const std::exception& e) {
                writeError(id, "internal_error", e.what());
            }
        }
        engine.shutdown();
        return 0;
    } catch (const mvdb::DbError& e) {
        std::cerr << "startup failed: " << e.code << ": " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "startup failed: " << e.what() << "\n";
        return 1;
    }
}
