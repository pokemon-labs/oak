// A minimal JSON reader (hoenn's tools/json.hpp), for the client's requests. Not
// part of the engine: it allocates, recurses and throws on malformed input.

#pragma once

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace json {

struct Value {
  enum Type : uint8_t { Null, Bool, Number, String, Array, Object };

  Type type = Null;
  bool boolean = false;
  double number = 0;
  std::string string;
  std::vector<Value> array;
  std::vector<std::pair<std::string, Value>> object;

  bool is_null() const { return type == Null; }

  // The member `key`, or a null value if there is none (or this isn't an
  // object).
  const Value& operator[](std::string_view key) const {
    static const Value null{};
    for (const auto& [k, v] : object) {
      if (k == key) return v;
    }
    return null;
  }
  const Value& operator[](size_t i) const { return array.at(i); }
  size_t size() const { return type == Array ? array.size() : object.size(); }

  int64_t integer() const { return int64_t(number); }
};

namespace detail {

struct Parser {
  std::string_view s;
  size_t i = 0;

  [[noreturn]] void fail(const char* what) const {
    throw std::runtime_error(std::string("json: ") + what + " at offset " +
                             std::to_string(i));
  }

  void ws() {
    while (i < s.size() &&
           (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) {
      i++;
    }
  }

  void expect(char c) {
    ws();
    if (i >= s.size() || s[i] != c) fail("unexpected character");
    i++;
  }

  bool literal(std::string_view word) {
    if (s.substr(i, word.size()) != word) return false;
    i += word.size();
    return true;
  }

  static void utf8(std::string& out, uint32_t cp) {
    if (cp < 0x80) {
      out += char(cp);
    } else if (cp < 0x800) {
      out += char(0xC0 | (cp >> 6));
      out += char(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
      out += char(0xE0 | (cp >> 12));
      out += char(0x80 | ((cp >> 6) & 0x3F));
      out += char(0x80 | (cp & 0x3F));
    } else {
      out += char(0xF0 | (cp >> 18));
      out += char(0x80 | ((cp >> 12) & 0x3F));
      out += char(0x80 | ((cp >> 6) & 0x3F));
      out += char(0x80 | (cp & 0x3F));
    }
  }

  uint32_t hex4() {
    if (i + 4 > s.size()) fail("short \\u escape");
    uint32_t v = 0;
    for (int k = 0; k < 4; k++) {
      const char c = s[i++];
      v <<= 4;
      if (c >= '0' && c <= '9') v |= uint32_t(c - '0');
      else if (c >= 'a' && c <= 'f') v |= uint32_t(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F') v |= uint32_t(c - 'A' + 10);
      else fail("bad \\u escape");
    }
    return v;
  }

  std::string str() {
    expect('"');
    std::string out;
    while (true) {
      if (i >= s.size()) fail("unterminated string");
      const char c = s[i++];
      if (c == '"') return out;
      if (c != '\\') {
        out += c;
        continue;
      }
      if (i >= s.size()) fail("unterminated escape");
      switch (s[i++]) {
      case '"': out += '"'; break;
      case '\\': out += '\\'; break;
      case '/': out += '/'; break;
      case 'b': out += '\b'; break;
      case 'f': out += '\f'; break;
      case 'n': out += '\n'; break;
      case 'r': out += '\r'; break;
      case 't': out += '\t'; break;
      case 'u': {
        uint32_t cp = hex4();
        if (cp >= 0xD800 && cp < 0xDC00 && literal("\\u")) {
          const uint32_t lo = hex4();
          cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
        }
        utf8(out, cp);
        break;
      }
      default: fail("bad escape");
      }
    }
  }

  Value value() {
    ws();
    if (i >= s.size()) fail("unexpected end");
    Value v;
    const char c = s[i];
    if (c == '{') {
      v.type = Value::Object;
      i++;
      ws();
      if (i < s.size() && s[i] == '}') {
        i++;
        return v;
      }
      while (true) {
        std::string key = str();
        expect(':');
        v.object.emplace_back(std::move(key), value());
        ws();
        if (i < s.size() && s[i] == ',') {
          i++;
          continue;
        }
        expect('}');
        return v;
      }
    }
    if (c == '[') {
      v.type = Value::Array;
      i++;
      ws();
      if (i < s.size() && s[i] == ']') {
        i++;
        return v;
      }
      while (true) {
        v.array.push_back(value());
        ws();
        if (i < s.size() && s[i] == ',') {
          i++;
          continue;
        }
        expect(']');
        return v;
      }
    }
    if (c == '"') {
      v.type = Value::String;
      v.string = str();
      return v;
    }
    if (literal("null")) return v;
    if (literal("true")) {
      v.type = Value::Bool;
      v.boolean = true;
      return v;
    }
    if (literal("false")) {
      v.type = Value::Bool;
      return v;
    }
    const std::string num(s.substr(i, s.find_first_of(",]} \t\r\n", i) - i));
    char* end = nullptr;
    v.type = Value::Number;
    v.number = std::strtod(num.c_str(), &end);
    if (num.empty() || end != num.c_str() + num.size()) fail("bad number");
    i += num.size();
    return v;
  }
};

} // namespace detail

inline Value parse(std::string_view text) {
  detail::Parser p{text};
  Value v = p.value();
  p.ws();
  if (p.i != text.size()) p.fail("trailing characters");
  return v;
}

} // namespace json
