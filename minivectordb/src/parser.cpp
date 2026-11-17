#include "mvdb.h"
#include "string_scan.h"

#include <cctype>
#include <cfloat>
#include <charconv>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <set>

namespace mvdb {

void fail(const std::string& code, const std::string& message) {
    throw DbError(code, message);
}

namespace {

std::string upperAscii(const std::string& s) {
    std::string out = s;
    for (char& c : out) {
        if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 'a' + 'A');
    }
    return out;
}

bool isIdentStart(unsigned char c) {
    return std::isalpha(c) || c == '_' || c == '$' || c >= 0x80;
}

bool isIdentChar(unsigned char c) {
    return std::isalnum(c) || c == '_' || c == '$' || c >= 0x80;
}

struct Token {
    enum Type { IDENT, NUMBER, STRING, PUNCT, END } type = END;
    std::string text;
    size_t pos = 0;
};

class Lexer {
public:
    explicit Lexer(const std::string& input) : s_(input) {}

    std::vector<Token> run() {
        std::vector<Token> out;
        for (;;) {
            Token t = next();
            if (t.type == Token::END) break;
            out.push_back(std::move(t));
        }
        out.push_back(Token{Token::END, "", s_.size()});
        return out;
    }

private:
    const std::string& s_;
    size_t i_ = 0;

    char cur() const { return i_ < s_.size() ? s_[i_] : '\0'; }
    char at(size_t off) const { return i_ + off < s_.size() ? s_[i_ + off] : '\0'; }

    void skipSpaceAndComments() {
        for (;;) {
            while (i_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[i_]))) ++i_;
            if (i_ + 1 < s_.size() && s_[i_] == '-' && s_[i_ + 1] == '-') {
                while (i_ < s_.size() && s_[i_] != '\n') ++i_;
                continue;
            }
            if (i_ < s_.size() && s_[i_] == '#') {
                while (i_ < s_.size() && s_[i_] != '\n') ++i_;
                continue;
            }
            if (i_ + 1 < s_.size() && s_[i_] == '/' && s_[i_ + 1] == '*') {
                i_ += 2;
                while (i_ + 1 < s_.size() && !(s_[i_] == '*' && s_[i_ + 1] == '/')) ++i_;
                if (i_ + 1 < s_.size()) i_ += 2;
                continue;
            }
            break;
        }
    }

    Token next() {
        skipSpaceAndComments();
        if (i_ >= s_.size()) return Token{Token::END, "", i_};
        size_t start = i_;
        char c = s_[i_];

        if (c == '\'' || c == '"') {
            char quote = c;
            ++i_;
            std::string value;
            while (i_ < s_.size()) {
                const size_t begin = i_;
                i_ = scanStringSpan(s_, i_, quote, false);
                value.append(s_, begin, i_ - begin);
                if (i_ == s_.size()) break;
                const char ch = s_[i_++];
                if (ch == quote) {
                    if (i_ < s_.size() && s_[i_] == quote) {  // doubled quote
                        value.push_back(quote);
                        ++i_;
                        continue;
                    }
                    return Token{Token::STRING, std::move(value), start};
                }
                if (ch == '\\' && i_ < s_.size()) {
                    char e = s_[i_++];
                    switch (e) {
                        case 'n': value.push_back('\n'); break;
                        case 't': value.push_back('\t'); break;
                        case 'r': value.push_back('\r'); break;
                        case '0': value.push_back('\0'); break;
                        case 'b': value.push_back('\b'); break;
                        case 'Z': value.push_back('\x1a'); break;
                        default: value.push_back(e); break;
                    }
                    continue;
                }
                value.push_back(ch);
            }
            fail("syntax_error", "unterminated string literal");
        }

        if (c == '`') {
            ++i_;
            std::string value;
            while (i_ < s_.size()) {
                char ch = s_[i_++];
                if (ch == '`') {
                    if (i_ < s_.size() && s_[i_] == '`') {
                        value.push_back('`');
                        ++i_;
                        continue;
                    }
                    return Token{Token::IDENT, value, start};
                }
                value.push_back(ch);
            }
            fail("syntax_error", "unterminated quoted identifier");
        }

        if (isIdentStart(static_cast<unsigned char>(c))) {
            ++i_;
            while (i_ < s_.size() && isIdentChar(static_cast<unsigned char>(s_[i_]))) ++i_;
            return Token{Token::IDENT, s_.substr(start, i_ - start), start};
        }

        bool signedNumber = (c == '-' || c == '+') &&
                            (std::isdigit(static_cast<unsigned char>(at(1))) || at(1) == '.');
        if (std::isdigit(static_cast<unsigned char>(c)) || (c == '.' && std::isdigit(static_cast<unsigned char>(at(1)))) ||
            signedNumber) {
            if (c == '-' || c == '+') ++i_;
            bool digits = false;
            while (i_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[i_]))) {
                ++i_;
                digits = true;
            }
            if (i_ < s_.size() && s_[i_] == '.') {
                ++i_;
                while (i_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[i_]))) {
                    ++i_;
                    digits = true;
                }
            }
            if (digits && (cur() == 'e' || cur() == 'E')) {
                size_t save = i_;
                ++i_;
                if (cur() == '+' || cur() == '-') ++i_;
                bool expDigits = false;
                while (i_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[i_]))) {
                    ++i_;
                    expDigits = true;
                }
                if (!expDigits) i_ = save;
            }
            if (digits) return Token{Token::NUMBER, s_.substr(start, i_ - start), start};
            i_ = start;
        }

        // punctuation / operators
        if (i_ + 1 < s_.size()) {
            std::string two = s_.substr(i_, 2);
            if (two == "<=" || two == ">=" || two == "<>" || two == "!=") {
                i_ += 2;
                return Token{Token::PUNCT, two, start};
            }
        }
        ++i_;
        return Token{Token::PUNCT, std::string(1, c), start};
    }
};

int64_t parseInt64Text(const std::string& text) {
    errno = 0;
    char* end = nullptr;
    long long value = std::strtoll(text.c_str(), &end, 10);
    if (errno == ERANGE || end == text.c_str() || *end != '\0') {
        fail("invalid_number", "integer literal out of range");
    }
    return static_cast<int64_t>(value);
}

double parseDoubleText(const std::string& text) {
    errno = 0;
    char* end = nullptr;
    double value = std::strtod(text.c_str(), &end);
    if (errno == ERANGE || end == text.c_str() || *end != '\0' || !std::isfinite(value)) {
        fail("invalid_number", "invalid numeric literal");
    }
    return value;
}

class Parser {
public:
    explicit Parser(const std::string& sql) : toks_(Lexer(sql).run()) {}

    Statement parse() {
        while (isPunct(";")) advance();
        if (peek().type == Token::END) fail("syntax_error", "empty SQL statement");

        std::string first = upper();
        if (first == "CREATE") return parseCreate();
        if (first == "INSERT") return parseInsert();
        if (first == "SELECT") return parseSelect();
        if (first == "UPDATE") return parseUpdate();
        if (first == "DELETE") return parseDelete();
        if (first == "BEGIN" || first == "START") return parseBegin();
        if (first == "COMMIT") return parseCommit();
        if (first == "ROLLBACK") return parseRollback();

        static const std::set<std::string> unsupported = {
            "ALTER", "DROP", "TRUNCATE", "SHOW", "DESCRIBE", "DESC", "USE", "SET",
            "EXPLAIN", "GRANT", "REVOKE", "ANALYZE", "OPTIMIZE", "RENAME", "REPLACE",
            "WITH", "CALL", "PREPARE", "EXECUTE", "DEALLOCATE", "LOCK", "UNLOCK",
            "FLUSH", "RESET", "SAVEPOINT", "RELEASE", "HANDLER", "LOAD", "COPY"};
        if (unsupported.count(first)) fail("unsupported_sql", "statement is not supported");
        fail("unsupported_sql", "unrecognized statement");
    }

private:
    std::vector<Token> toks_;
    size_t i_ = 0;

    const Token& peek(size_t off = 0) const {
        size_t idx = i_ + off;
        if (idx >= toks_.size()) return toks_.back();
        return toks_[idx];
    }
    const Token& cur() const { return peek(); }
    std::string upper(size_t off = 0) const { return upperAscii(peek(off).text); }
    bool isKw(const char* kw, size_t off = 0) const { return upper(off) == kw; }
    bool isPunct(const char* p, size_t off = 0) const {
        const Token& t = peek(off);
        return t.type == Token::PUNCT && t.text == p;
    }
    const Token& advance() {
        const Token& t = peek();
        if (i_ < toks_.size() - 1) ++i_;
        return t;
    }
    bool acceptKw(const char* kw) {
        if (isKw(kw)) { advance(); return true; }
        return false;
    }
    bool acceptPunct(const char* p) {
        if (isPunct(p)) { advance(); return true; }
        return false;
    }
    void expectKw(const char* kw) {
        if (!acceptKw(kw)) fail("syntax_error", std::string("expected ") + kw);
    }
    void expectPunct(const char* p) {
        if (!acceptPunct(p)) fail("syntax_error", std::string("expected ") + p);
    }
    std::string expectIdent() {
        const Token& t = peek();
        if (t.type != Token::IDENT) fail("syntax_error", "expected identifier");
        std::string name = t.text;
        advance();
        if (name.empty()) fail("syntax_error", "empty identifier");
        return name;
    }
    void endStatement() {
        while (isPunct(";")) advance();
        if (peek().type != Token::END) {
            std::string u = upper();
            if (u == "JOIN" || u == "OR" || u == "GROUP" || u == "HAVING" || u == "UNION" ||
                u == "DISTINCT" || u == "IS" || u == "LIKE" || u == "IN") {
                fail("unsupported_sql", "SQL feature is not supported");
            }
            fail("syntax_error", "unexpected token after statement");
        }
    }

    Literal parseLiteral(bool allowVector) {
        const Token& t = peek();
        Literal lit;
        if (t.type == Token::STRING) {
            lit.kind = Literal::STR;
            lit.s = t.text;
            advance();
            return lit;
        }
        if (t.type == Token::NUMBER) {
            std::string text = t.text;
            advance();
            if (text.find('.') != std::string::npos || text.find('e') != std::string::npos ||
                text.find('E') != std::string::npos) {
                lit.kind = Literal::DOUBLE;
                lit.d = parseDoubleText(text);
            } else {
                lit.kind = Literal::INT;
                lit.i = parseInt64Text(text);
            }
            return lit;
        }
        if (isPunct("[")) {
            if (!allowVector) fail("unsupported_sql", "vector literal is not allowed here");
            lit.kind = Literal::VECLIT;
            lit.v = parseVectorToken();
            return lit;
        }
        if (isKw("NULL")) {
            advance();
            lit.kind = Literal::NULLV;
            return lit;
        }
        if (isPunct("-") || isPunct("+")) {
            fail("syntax_error", "invalid numeric literal");
        }
        fail("syntax_error", "expected literal");
    }

    std::vector<double> parseVectorToken() {
        expectPunct("[");
        std::vector<double> out;
        if (acceptPunct("]")) return out;
        for (;;) {
            const Token& t = peek();
            if (t.type != Token::NUMBER) fail("invalid_vector", "vector literal must contain numbers");
            std::string text = t.text;
            advance();
            double d = 0;
            try {
                d = parseDoubleText(text);
            } catch (const DbError&) {
                fail("invalid_vector", "invalid vector component");
            }
            if (!std::isfinite(d)) fail("invalid_vector", "vector component is not finite");
            out.push_back(d);
            if (acceptPunct(",")) continue;
            expectPunct("]");
            break;
        }
        return out;
    }

    Statement parseCreate() {
        expectKw("CREATE");
        Statement st;
        if (acceptKw("TABLE")) {
            st.kind = StmtKind::CREATE_TABLE;
            st.createTable = parseCreateTableBody();
            endStatement();
            return st;
        }
        if (acceptKw("VECTOR")) {
            expectKw("INDEX");
            st.kind = StmtKind::CREATE_INDEX;
            st.createIndex = parseCreateIndexBody();
            endStatement();
            return st;
        }
        fail("unsupported_sql", "only CREATE TABLE and CREATE VECTOR INDEX are supported");
    }

    CreateTableStmt parseCreateTableBody() {
        CreateTableStmt st;
        st.table = expectIdent();
        expectPunct("(");
        if (isPunct(")")) fail("syntax_error", "table must have columns");
        for (;;) {
            if (isKw("PRIMARY")) {
                advance();
                expectKw("KEY");
                expectPunct("(");
                std::string pk = expectIdent();
                expectPunct(")");
                if (!st.tablePk.empty()) fail("syntax_error", "duplicate primary key definition");
                st.tablePk = pk;
            } else {
                ColumnDef def;
                def.name = expectIdent();
                std::string type = upper();
                if (type == "BIGINT") {
                    advance();
                    def.type = ColType::BIGINT;
                } else if (type == "VARCHAR") {
                    advance();
                    expectPunct("(");
                    const Token& n = peek();
                    if (n.type != Token::NUMBER) fail("syntax_error", "VARCHAR length must be an integer");
                    int64_t len = parseInt64Text(n.text);
                    advance();
                    expectPunct(")");
                    if (len <= 0 || len > 1000000) fail("invalid_schema", "invalid VARCHAR length");
                    def.type = ColType::VARCHAR;
                    def.length = static_cast<uint32_t>(len);
                } else if (type == "VECTOR") {
                    advance();
                    expectPunct("(");
                    const Token& n = peek();
                    if (n.type != Token::NUMBER) fail("syntax_error", "VECTOR dimension must be an integer");
                    int64_t dim = parseInt64Text(n.text);
                    advance();
                    expectPunct(")");
                    if (dim <= 0 || dim > 1000000) fail("invalid_schema", "invalid VECTOR dimension");
                    def.type = ColType::VECTOR;
                    def.length = static_cast<uint32_t>(dim);
                } else {
                    fail("unsupported_sql", "unsupported column type");
                }
                while (true) {
                    if (isKw("PRIMARY")) {
                        advance();
                        expectKw("KEY");
                        def.primary = true;
                    } else if (isKw("NOT")) {
                        advance();
                        expectKw("NULL");
                    } else if (isKw("NULL")) {
                        advance();
                    } else {
                        break;
                    }
                }
                st.columns.push_back(std::move(def));
            }
            if (acceptPunct(",")) continue;
            expectPunct(")");
            break;
        }
        return st;
    }

    CreateIndexStmt parseCreateIndexBody() {
        CreateIndexStmt st;
        st.index = expectIdent();
        expectKw("ON");
        st.table = expectIdent();
        expectPunct("(");
        st.column = expectIdent();
        expectPunct(")");
        return st;
    }

    Statement parseInsert() {
        expectKw("INSERT");
        acceptKw("INTO");
        Statement st;
        st.kind = StmtKind::INSERT;
        st.insert.table = expectIdent();
        if (acceptPunct("(")) {
            st.insert.hasColumns = true;
            for (;;) {
                st.insert.columns.push_back(expectIdent());
                if (acceptPunct(",")) continue;
                expectPunct(")");
                break;
            }
        }
        expectKw("VALUES");
        for (;;) {
            expectPunct("(");
            std::vector<Literal> row;
            if (isPunct(")")) fail("syntax_error", "empty VALUES row");
            for (;;) {
                row.push_back(parseLiteral(true));
                if (acceptPunct(",")) continue;
                expectPunct(")");
                break;
            }
            st.insert.rows.push_back(std::move(row));
            if (acceptPunct(",")) continue;
            break;
        }
        endStatement();
        return st;
    }

    Statement parseSelect() {
        expectKw("SELECT");
        Statement st;
        st.kind = StmtKind::SELECT;
        if (isKw("DISTINCT")) fail("unsupported_sql", "DISTINCT is not supported");
        for (;;) {
            SelectItem item;
            if (acceptPunct("*")) {
                item.kind = SelectItem::STAR;
            } else if (isKw("L2_DISTANCE") && isPunct("(", 1)) {
                advance();
                expectPunct("(");
                item.kind = SelectItem::DIST;
                item.column = expectIdent();
                expectPunct(",");
                item.vec = parseVectorArgument(item.vecText);
                expectPunct(")");
            } else {
                item.kind = SelectItem::COLUMN;
                item.column = expectIdent();
            }
            if (acceptKw("AS")) {
                item.alias = expectIdent();
                item.hasAlias = true;
            } else if (peek().type == Token::IDENT) {
                std::string u = upper();
                if (u != "FROM" && u != "WHERE" && u != "ORDER" && u != "LIMIT" && u != "AND") {
                    item.alias = expectIdent();
                    item.hasAlias = true;
                }
            }
            st.select.items.push_back(std::move(item));
            if (acceptPunct(",")) continue;
            break;
        }
        expectKw("FROM");
        st.select.table = expectIdent();
        if (isKw("JOIN") || isKw("OR") || isKw("GROUP") || isKw("UNION") || isKw("HAVING")) {
            fail("unsupported_sql", "SQL feature is not supported");
        }
        if (acceptKw("WHERE")) {
            st.select.conds = parseConditions();
        }
        if (acceptKw("ORDER")) {
            expectKw("BY");
            for (;;) {
                OrderItem item;
                if (isKw("L2_DISTANCE") && isPunct("(", 1)) {
                    advance();
                    expectPunct("(");
                    item.kind = OrderItem::DIST;
                    item.name = expectIdent();
                    expectPunct(",");
                    item.vec = parseVectorArgument(item.vecText);
                    expectPunct(")");
                } else {
                    item.kind = OrderItem::COLUMN;
                    item.name = expectIdent();
                }
                if (acceptKw("ASC")) {
                    item.desc = false;
                } else if (acceptKw("DESC")) {
                    item.desc = true;
                }
                st.select.order.push_back(std::move(item));
                if (acceptPunct(",")) continue;
                break;
            }
        }
        if (acceptKw("LIMIT")) {
            const Token& n = peek();
            if (n.type != Token::NUMBER) fail("syntax_error", "LIMIT requires an integer");
            int64_t value = parseInt64Text(n.text);
            advance();
            if (value < 0) fail("syntax_error", "LIMIT must be non-negative");
            st.select.hasLimit = true;
            st.select.limit = value;
        }
        endStatement();
        return st;
    }

    std::vector<double> parseVectorArgument(std::string& text) {
        if (peek().type == Token::STRING) {
            std::string raw = peek().text;
            text = "'" + raw + "'";
            advance();
            return parseVectorText(raw);
        }
        if (isPunct("[")) {
            std::vector<double> v = parseVectorToken();
            text = "[...]";
            return v;
        }
        fail("syntax_error", "expected vector literal");
    }

    std::vector<Cmp> parseConditions() {
        std::vector<Cmp> out;
        parseConditionGroup(out);
        return out;
    }

    void parseConditionGroup(std::vector<Cmp>& out) {
        for (;;) {
            if (isPunct("(")) {
                advance();
                parseConditionGroup(out);
                expectPunct(")");
            } else {
                out.push_back(parseComparison());
            }
            if (acceptKw("AND")) continue;
            if (isKw("OR") || isKw("XOR")) fail("unsupported_sql", "OR is not supported");
            break;
        }
    }

    Cmp parseComparison() {
        struct Operand {
            bool isColumn = false;
            std::string column;
            Literal lit;
        };
        auto parseOperand = [&]() -> Operand {
            Operand o;
            if (peek().type == Token::IDENT) {
                std::string u = upper();
                if (u == "NULL" || u == "TRUE" || u == "FALSE") {
                    fail("unsupported_sql", "NULL and boolean literals are not supported");
                }
                o.isColumn = true;
                o.column = expectIdent();
            } else {
                o.lit = parseLiteral(false);
            }
            return o;
        };
        Operand left = parseOperand();
        const Token& opTok = peek();
        int op = -1;
        if (opTok.type == Token::PUNCT) {
            if (opTok.text == "=") op = 0;
            else if (opTok.text == "<") op = 1;
            else if (opTok.text == "<=") op = 2;
            else if (opTok.text == ">") op = 3;
            else if (opTok.text == ">=") op = 4;
            else if (opTok.text == "<>" || opTok.text == "!=") op = 5;
        }
        if (op < 0) fail("unsupported_sql", "unsupported comparison operator");
        advance();
        Operand right = parseOperand();
        if (left.isColumn == right.isColumn) {
            fail("unsupported_sql", "comparison requires one column and one literal");
        }
        Cmp cmp;
        cmp.op = op;
        if (left.isColumn) {
            cmp.column = left.column;
            cmp.lit = right.lit;
            cmp.litOnLeft = false;
        } else {
            cmp.column = right.column;
            cmp.lit = left.lit;
            cmp.litOnLeft = true;
        }
        if (cmp.lit.kind == Literal::VECLIT) {
            fail("unsupported_sql", "vector comparison is not supported");
        }
        return cmp;
    }

    Statement parseUpdate() {
        expectKw("UPDATE");
        Statement st;
        st.kind = StmtKind::UPDATE;
        st.update.table = expectIdent();
        expectKw("SET");
        for (;;) {
            Assignment a;
            a.column = expectIdent();
            expectPunct("=");
            a.lit = parseLiteral(true);
            st.update.sets.push_back(std::move(a));
            if (acceptPunct(",")) continue;
            break;
        }
        if (acceptKw("WHERE")) {
            st.update.hasWhere = true;
            st.update.conds = parseConditions();
        }
        endStatement();
        return st;
    }

    Statement parseDelete() {
        expectKw("DELETE");
        expectKw("FROM");
        Statement st;
        st.kind = StmtKind::DELETE;
        st.del.table = expectIdent();
        if (acceptKw("WHERE")) {
            st.del.hasWhere = true;
            st.del.conds = parseConditions();
        }
        endStatement();
        return st;
    }

    Statement parseBegin() {
        if (isKw("START")) {
            advance();
            expectKw("TRANSACTION");
        } else {
            expectKw("BEGIN");
            if (!acceptKw("WORK")) acceptKw("TRANSACTION");
        }
        endStatement();
        Statement st;
        st.kind = StmtKind::BEGIN;
        return st;
    }

    Statement parseCommit() {
        expectKw("COMMIT");
        if (!acceptKw("WORK")) acceptKw("TRANSACTION");
        endStatement();
        Statement st;
        st.kind = StmtKind::COMMIT;
        return st;
    }

    Statement parseRollback() {
        expectKw("ROLLBACK");
        if (!acceptKw("WORK")) acceptKw("TRANSACTION");
        endStatement();
        Statement st;
        st.kind = StmtKind::ROLLBACK;
        return st;
    }
};

}  // namespace

std::vector<double> parseVectorText(const std::string& text, size_t expectedSize) {
    std::vector<double> out;
    out.reserve(expectedSize < text.size() / 2 ? expectedSize : text.size() / 2);
    size_t i = 0;
    auto skipws = [&]() {
        while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i]))) ++i;
    };
    skipws();
    if (i >= text.size() || text[i] != '[') fail("invalid_vector", "vector literal must start with '['");
    ++i;
    skipws();
    if (i < text.size() && text[i] == ']') {
        ++i;
        skipws();
        if (i != text.size()) fail("invalid_vector", "trailing characters after vector literal");
        return out;
    }
    for (;;) {
        skipws();
        if (i >= text.size()) fail("invalid_vector", "unterminated vector literal");
        const char* begin = text.data() + i;
        double d = 0;
        const auto parsed = std::from_chars(begin, text.data() + text.size(), d);
        const char* end = parsed.ptr;
        // Keep the original conversion semantics for plus signs, hexadecimal
        // literals and range boundaries.
        if (parsed.ec != std::errc{} ||
            (end < text.data() + text.size() && (*end == 'x' || *end == 'X')) ||
            (d != 0 && std::fabs(d) < std::numeric_limits<double>::min())) {
            char* fallbackEnd = nullptr;
            errno = 0;
            d = std::strtod(begin, &fallbackEnd);
            if (fallbackEnd == begin || errno == ERANGE) {
                fail("invalid_vector", "invalid vector component");
            }
            end = fallbackEnd;
        }
        if (!std::isfinite(d)) fail("invalid_vector", "vector component must be finite");
        out.push_back(d);
        i = static_cast<size_t>(end - text.data());
        skipws();
        if (i < text.size() && text[i] == ',') {
            ++i;
            continue;
        }
        if (i < text.size() && text[i] == ']') {
            ++i;
            skipws();
            if (i != text.size()) fail("invalid_vector", "trailing characters after vector literal");
            break;
        }
        fail("invalid_vector", "malformed vector literal");
    }
    return out;
}

// Fast path for the plain decimal form [-]<digits>[.<digits>] with at most 18
// significant digits and no exponent.  The quotient of two exactly
// representable integers is correctly rounded, so mantissa / 10^fraction
// equals the value std::from_chars would return; anything else falls back to
// the general path.
bool parsePlainDecimal(const char* begin, const char* end, const char*& outEnd, double& out) {
    const char* p = begin;
    bool negative = false;
    if (p < end && (*p == '-' || *p == '+')) {
        negative = (*p == '-');
        ++p;
    }
    uint64_t mantissa = 0;
    int digits = 0;
    int fraction = 0;
    bool any = false;
    for (; p < end && *p >= '0' && *p <= '9'; ++p) {
        if (digits >= 18) return false;
        mantissa = mantissa * 10 + static_cast<uint64_t>(*p - '0');
        ++digits;
        any = true;
    }
    if (p < end && *p == '.') {
        ++p;
        for (; p < end && *p >= '0' && *p <= '9'; ++p) {
            if (digits >= 18) return false;
            mantissa = mantissa * 10 + static_cast<uint64_t>(*p - '0');
            ++digits;
            ++fraction;
            any = true;
        }
    }
    if (!any) return false;
    // Reject anything that would need the general parser ('e', 'E', 'x', ...).
    if (p < end && *p != ',' && *p != ']' && !std::isspace(static_cast<unsigned char>(*p))) return false;
    static const double kPow10[19] = {1e0,  1e1,  1e2,  1e3,  1e4,  1e5,  1e6,  1e7,  1e8,  1e9,
                                      1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18};
    double value = static_cast<double>(mantissa) / kPow10[fraction];
    out = negative ? -value : value;
    outEnd = p;
    return true;
}

// Same grammar and error codes as parseVectorText, but converts each component
// to float32 in place and validates the float range while parsing.  The row
// path uses it to avoid a throw-away vector<double> per row.
std::vector<float> parseVectorTextFloat(const std::string& text, size_t expectedSize) {
    std::vector<float> out;
    out.reserve(expectedSize < text.size() / 2 ? expectedSize : text.size() / 2);
    size_t i = 0;
    auto skipws = [&]() {
        while (i < text.size() && std::isspace(static_cast<unsigned char>(text[i]))) ++i;
    };
    skipws();
    if (i >= text.size() || text[i] != '[') fail("invalid_vector", "vector literal must start with '['");
    ++i;
    skipws();
    if (i < text.size() && text[i] == ']') {
        ++i;
        skipws();
        if (i != text.size()) fail("invalid_vector", "trailing characters after vector literal");
        return out;
    }
    for (;;) {
        skipws();
        if (i >= text.size()) fail("invalid_vector", "unterminated vector literal");
        const char* begin = text.data() + i;
        double d = 0;
        const char* end = begin;
        if (!parsePlainDecimal(begin, text.data() + text.size(), end, d)) {
            const auto parsed = std::from_chars(begin, text.data() + text.size(), d);
            end = parsed.ptr;
            if (parsed.ec != std::errc{} ||
                (end < text.data() + text.size() && (*end == 'x' || *end == 'X')) ||
                (d != 0 && std::fabs(d) < std::numeric_limits<double>::min())) {
                char* fallbackEnd = nullptr;
                errno = 0;
                d = std::strtod(begin, &fallbackEnd);
                if (fallbackEnd == begin || errno == ERANGE) {
                    fail("invalid_vector", "invalid vector component");
                }
                end = fallbackEnd;
            }
        }
        if (!std::isfinite(d)) fail("invalid_vector", "vector component must be finite");
        if (std::fabs(d) > static_cast<double>(FLT_MAX)) {
            fail("invalid_vector", "vector component is not a finite float32");
        }
        const float f = static_cast<float>(d);
        if (!std::isfinite(f)) fail("invalid_vector", "vector component overflows float32");
        out.push_back(f);
        i = static_cast<size_t>(end - text.data());
        skipws();
        if (i < text.size() && text[i] == ',') {
            ++i;
            continue;
        }
        if (i < text.size() && text[i] == ']') {
            ++i;
            skipws();
            if (i != text.size()) fail("invalid_vector", "trailing characters after vector literal");
            break;
        }
        fail("invalid_vector", "malformed vector literal");
    }
    return out;
}

Statement parseStatement(const std::string& sql) {
    Parser parser(sql);
    return parser.parse();
}

}  // namespace mvdb
