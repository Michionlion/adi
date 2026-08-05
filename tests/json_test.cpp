#include "adi/json.hpp"

#include <cassert>
#include <string>
#include <stdexcept>

int main() {
    const auto value = adi::parse_json(
        R"({"input":"hello\nworld","max":2,"stream":false,"items":[null,"\u263a"]})");
    assert(value.find("input")->string() != nullptr);
    assert(*value.find("max")->number() == 2.0);
    assert(*value.find("stream")->boolean() == false);
    assert(value.find("items")->array()->size() == 2);
    assert(adi::json_string("a\n\"b") == R"("a\n\"b")");

    bool rejected = false;
    try {
        (void)adi::parse_json(R"({"a":1,"a":2})");
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    assert(rejected);

    for (const auto malformed : {
             std::string{"\"\xC0\x80\"", 4},
             std::string{"\"\xED\xA0\x80\"", 5},
             std::string{R"("\uDC00")"},
         }) {
        rejected = false;
        try {
            (void)adi::parse_json(malformed);
        } catch (const std::runtime_error &) {
            rejected = true;
        }
        assert(rejected);
    }

    std::string nested = "0";
    for (int index = 0; index < 129; ++index) {
        nested = "[" + nested + "]";
    }
    rejected = false;
    try {
        (void)adi::parse_json(nested);
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    assert(rejected);

    assert(adi::json_string(std::string{"x\xFFy", 3}) == "\"x\xEF\xBF\xBDy\"");
}
