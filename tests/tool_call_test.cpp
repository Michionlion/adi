#include "adi/json.hpp"
#include "tool_call.hpp"

#include <cassert>
#include <stdexcept>
#include <string>

int main() {
    const auto tools = adi::parse_function_tools(adi::parse_json(R"([
        {"type":"function","name":"calculate","description":"Arithmetic",
         "parameters":{"type":"object","properties":{
             "operation":{"type":"string"},"a":{"type":"number"},
             "b":{"type":"number"},"round":{"type":"boolean"}},
         "required":["operation","a","b"],"additionalProperties":false}}
    ])"));
    assert(tools.size() == 1);
    assert(tools[0].name == "calculate");
    assert(*tools[0].prompt_definition.find("type")->string() == "function");
    assert(*tools[0].prompt_definition.find("function")
                ->find("name")->string() == "calculate");
    assert(*tools[0].response_definition.find("strict")->boolean() == false);

    const auto parsed = adi::parse_tool_calls(
        "I should calculate this.\n<tool_call>\n<function=calculate>\n"
        "<parameter=operation>\nadd\n</parameter>\n"
        "<parameter=a>\n2.5\n</parameter>\n"
        "<parameter=b>\n4\n</parameter>\n"
        "<parameter=round>\nfalse\n</parameter>\n"
        "</function>\n</tool_call>",
        tools,
        true);
    assert(parsed.text == "I should calculate this.\n\n</think>");
    assert(parsed.function_calls.size() == 1);
    assert(parsed.function_calls[0].name == "calculate");
    assert(parsed.function_calls[0].arguments_json ==
           R"({"operation":"add","a":2.5,"b":4,"round":false})");

    bool rejected = false;
    try {
        (void)adi::parse_tool_calls(
            "<tool_call><function=calculate>"
            "<parameter=operation>add</parameter>"
            "<parameter=a>2</parameter>"
            "</function></tool_call>",
            tools,
            true);
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    assert(rejected);

    rejected = false;
    try {
        (void)adi::parse_function_tools(adi::parse_json(
            R"([{"type":"function","name":"bad","strict":true}])"));
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    assert(rejected);
}
