#pragma once

#include <cstdint>

namespace adi {

// Execution tuning that never changes model semantics. The prefill microbatch
// is the number of prompt tokens that share one packed-weight decode pass, so
// it trades scratch memory for prompt throughput. It is deliberately not part
// of the request schema: two requests with the same prompt must produce the
// same tokens regardless of how the server was launched.
struct ExecutionOptions {
    static constexpr std::uint32_t minimum_prefill_ubatch = 1;
    static constexpr std::uint32_t maximum_prefill_ubatch = 4096;

    std::uint32_t prefill_ubatch = 64;
};

// Throws std::invalid_argument when a field is outside its supported range.
void validate_execution_options(const ExecutionOptions &options);

} // namespace adi
