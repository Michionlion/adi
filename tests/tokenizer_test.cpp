#include "adi/model.hpp"
#include "adi/tokenizer.hpp"

#include <cassert>
#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>

int main(int argc, char **argv) {
    assert(argc == 2);
    adi::MachModel model(argv[1]);
    adi::Tokenizer tokenizer(model);

    const std::vector<std::pair<std::string_view, std::vector<std::uint32_t>>>
        golden{
            {"foo  bar", {7724, 220, 3498}},
            {"a\xF0\x9F\x99\x82" "b", {64, 169171, 65}},
            {"e\xCC\x81", {933}},
            {"\xD9\x85\xD8\xB1\xD8\xAD\xD8\xA8\xD8\xA7 "
             "\xD9\xA1\xD9\xA2\xD9\xA3",
             {148739, 28850, 150027, 220, 149, 94, 149, 95, 149, 96}},
            {"\xE4\xBD\xA0\xE5\xA5\xBD\xEF\xBC\x8C"
             "\xE4\xB8\x96\xE7\x95\x8C\xEF\xBC\x81",
             {109266, 3709, 96748, 6115}},
            {"<|im_start|>user\nhi<|im_end|>",
             {248045, 846, 198, 5834, 248046}},
            {"\xF0\x90\x97\x92\xCC\x87",
             {172, 238, 245, 240, 136, 229}},
            {"\xF0\x91\xA4\xB5\xF0\x91\xA4\xB0",
             {172, 239, 97, 113, 172, 239, 97, 108}},
        };
    for (const auto &[text, expected] : golden) {
        assert(tokenizer.encode(text) == expected);
    }
    assert(tokenizer.encode("e\xCC\x81") == tokenizer.encode("\xC3\xA9"));

    const auto tokens =
        model.gguf().string_array("tokenizer.ggml.tokens");
    const auto types =
        model.gguf().integer_array("tokenizer.ggml.token_type");
    for (std::uint32_t token = 0; token < tokens.size(); ++token) {
        if (types[token] == 3 || types[token] == 4) {
            assert(tokenizer.encode(tokens[token]) ==
                   std::vector<std::uint32_t>{token});
        }
    }
}
