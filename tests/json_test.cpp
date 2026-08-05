#include "adi/json.hpp"

#include <cassert>
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
}
