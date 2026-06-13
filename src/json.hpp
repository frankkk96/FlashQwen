// Minimal recursive-descent JSON parser (header-only). No external deps.
// Just enough to parse model config, the safetensors header, and vocab.json.
// Strings are decoded to UTF-8 (handles \uXXXX, including surrogate pairs).
#pragma once
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <stdexcept>
#include <cstdint>

namespace minijson {

struct Value;
using ValuePtr = std::shared_ptr<Value>;

enum class Type { Null, Bool, Number, String, Array, Object };

struct Value {
    Type type = Type::Null;
    bool boolean = false;
    double number = 0.0;
    std::string str;                              // UTF-8 decoded
    std::vector<ValuePtr> arr;
    std::map<std::string, ValuePtr> obj;          // insertion order not preserved

    bool is_obj() const { return type == Type::Object; }
    bool is_arr() const { return type == Type::Array; }
    bool contains(const std::string& k) const { return obj.count(k) > 0; }
    const Value& operator[](const std::string& k) const { return *obj.at(k); }
    const Value& operator[](size_t i) const { return *arr.at(i); }
    int as_int() const { return (int)number; }
    long as_long() const { return (long)number; }
    double as_double() const { return number; }
    const std::string& as_str() const { return str; }
};

class Parser {
public:
    explicit Parser(const char* data, size_t n) : p_(data), end_(data + n) {}
    explicit Parser(const std::string& s) : p_(s.data()), end_(s.data() + s.size()) {}

    ValuePtr parse() {
        skip_ws();
        ValuePtr v = parse_value();
        return v;
    }

private:
    const char* p_;
    const char* end_;

    [[noreturn]] void fail(const std::string& msg) { throw std::runtime_error("JSON parse error: " + msg); }
    void skip_ws() { while (p_ < end_ && (*p_==' '||*p_=='\t'||*p_=='\n'||*p_=='\r')) ++p_; }
    char peek() { return p_ < end_ ? *p_ : '\0'; }

    ValuePtr parse_value() {
        skip_ws();
        char c = peek();
        switch (c) {
            case '{': return parse_object();
            case '[': return parse_array();
            case '"': { auto v = std::make_shared<Value>(); v->type = Type::String; v->str = parse_string(); return v; }
            case 't': case 'f': return parse_bool();
            case 'n': return parse_null();
            default: return parse_number();
        }
    }

    ValuePtr parse_object() {
        auto v = std::make_shared<Value>(); v->type = Type::Object;
        ++p_; // {
        skip_ws();
        if (peek() == '}') { ++p_; return v; }
        while (true) {
            skip_ws();
            if (peek() != '"') fail("expected key string");
            std::string key = parse_string();
            skip_ws();
            if (peek() != ':') fail("expected ':'");
            ++p_;
            v->obj[key] = parse_value();
            skip_ws();
            char c = peek();
            if (c == ',') { ++p_; continue; }
            if (c == '}') { ++p_; break; }
            fail("expected ',' or '}'");
        }
        return v;
    }

    ValuePtr parse_array() {
        auto v = std::make_shared<Value>(); v->type = Type::Array;
        ++p_; // [
        skip_ws();
        if (peek() == ']') { ++p_; return v; }
        while (true) {
            v->arr.push_back(parse_value());
            skip_ws();
            char c = peek();
            if (c == ',') { ++p_; continue; }
            if (c == ']') { ++p_; break; }
            fail("expected ',' or ']'");
        }
        return v;
    }

    static void append_utf8(std::string& out, uint32_t cp) {
        if (cp <= 0x7F) out.push_back((char)cp);
        else if (cp <= 0x7FF) {
            out.push_back((char)(0xC0 | (cp >> 6)));
            out.push_back((char)(0x80 | (cp & 0x3F)));
        } else if (cp <= 0xFFFF) {
            out.push_back((char)(0xE0 | (cp >> 12)));
            out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back((char)(0x80 | (cp & 0x3F)));
        } else {
            out.push_back((char)(0xF0 | (cp >> 18)));
            out.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back((char)(0x80 | (cp & 0x3F)));
        }
    }

    uint32_t parse_hex4() {
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i) {
            char c = *p_++;
            v <<= 4;
            if (c >= '0' && c <= '9') v |= (c - '0');
            else if (c >= 'a' && c <= 'f') v |= (c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= (c - 'A' + 10);
            else fail("bad hex digit");
        }
        return v;
    }

    std::string parse_string() {
        std::string out;
        ++p_; // opening quote
        while (p_ < end_) {
            char c = *p_++;
            if (c == '"') return out;
            if (c == '\\') {
                char e = *p_++;
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
                        uint32_t cp = parse_hex4();
                        if (cp >= 0xD800 && cp <= 0xDBFF) { // high surrogate
                            if (p_ + 1 < end_ && p_[0]=='\\' && p_[1]=='u') {
                                p_ += 2;
                                uint32_t lo = parse_hex4();
                                cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                            }
                        }
                        append_utf8(out, cp);
                        break;
                    }
                    default: fail("bad escape");
                }
            } else {
                out.push_back(c);
            }
        }
        fail("unterminated string");
    }

    ValuePtr parse_bool() {
        auto v = std::make_shared<Value>(); v->type = Type::Bool;
        if (end_ - p_ >= 4 && std::string(p_,4)=="true") { v->boolean = true; p_ += 4; }
        else if (end_ - p_ >= 5 && std::string(p_,5)=="false") { v->boolean = false; p_ += 5; }
        else fail("bad bool");
        return v;
    }

    ValuePtr parse_null() {
        auto v = std::make_shared<Value>(); v->type = Type::Null;
        if (end_ - p_ >= 4 && std::string(p_,4)=="null") p_ += 4; else fail("bad null");
        return v;
    }

    ValuePtr parse_number() {
        auto v = std::make_shared<Value>(); v->type = Type::Number;
        const char* start = p_;
        if (peek()=='-'||peek()=='+') ++p_;
        while (p_ < end_ && ((*p_>='0'&&*p_<='9')||*p_=='.'||*p_=='e'||*p_=='E'||*p_=='-'||*p_=='+')) ++p_;
        v->number = std::strtod(std::string(start, p_-start).c_str(), nullptr);
        return v;
    }
};

inline ValuePtr parse(const std::string& s) { Parser p(s); return p.parse(); }
inline ValuePtr parse(const char* data, size_t n) { Parser p(data, n); return p.parse(); }

} // namespace minijson
