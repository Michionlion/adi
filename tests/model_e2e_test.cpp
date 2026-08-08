#include "adi/generation.hpp"
#include "adi/model.hpp"
#include "adi/tokenizer.hpp"
#include "model_test_main.hpp"

#include <cassert>
#include <string>
#include <string_view>
#include <thread>

int ADI_MODEL_TEST_MAIN(
    int argc,
    adi::test_detail::CommandCharacter **argv) {
    assert(argc == 2);
    adi::MachModel model(adi::test_detail::model_path(argv[1]));
    adi::Tokenizer tokenizer(model);
    adi::GenerationOptions options;
    options.max_output_tokens = 1;
    options.temperature = 0.0F;
    options.top_p = 1.0F;
    adi::ContinuousBatcher batcher(model, tokenizer);
    const auto caller = std::this_thread::get_id();
    std::thread::id callback_thread;
    std::string streamed;
    const auto result = batcher.generate_from_prompt(
        adi::qwen_user_prompt("Hi"),
        options,
        [&](std::string_view piece) {
            callback_thread = std::this_thread::get_id();
            streamed.append(piece);
        });
    assert(result.text == "Here");
    assert(streamed == result.text);
    assert(callback_thread == caller);
    assert(result.input_tokens == 11);
    assert(result.output_tokens == 1);
    assert(result.finish_reason == adi::FinishReason::length);
    assert(result.prefill_seconds > 0.0);
    assert(result.decode_seconds > 0.0);
}
