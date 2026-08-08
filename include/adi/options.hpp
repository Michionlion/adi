#pragma once

#include <cstdint>

namespace adi {

// Execution tuning that never changes model semantics. The prefill microbatch
// is the number of prompt tokens that share one packed-weight decode pass. It
// is deliberately not part of the request schema: two requests with the same
// prompt must produce the same tokens regardless of how the server was
// launched.
//
// The default is measured, not assumed. It was sixteen for as long as the
// expert codec penalized a wide microbatch: mach_expert_matmul's cost per row
// grew with the batch, so raising the microbatch made the dominant kernel
// slower faster than it made anything else faster. Chunking expert work
// removed part of that penalty and a batch-lane expert kernel removed the
// rest, so throughput is now monotonic in the microbatch.
//
// On eight EPYC 9645 cores, medians of seven runs, a 1024-token prompt:
//
//   microbatch       64     128     256     512    1024
//   tokens/second   25.6    29.3    32.4    34.7    36.1
//   peak scratch    31.8    63.5   126.9   253.6   507.2 MB
//
// Nothing above is a falloff, so the default is an exchange rate rather than
// an optimum. Scratch doubles at every step while the gain shrinks: 128 buys
// 14.7% over 64, and every doubling after that buys at most 10.5%. That makes
// 128 the last step worth taking by default.
//
// Scratch is only ever allocated for the microbatch actually used, which is
// min(this, prompt length), so a short prompt costs the same whatever this is
// set to; only a caller sending long prompts pays for a large value. Callers
// with the memory to spare raise it with --ubatch, and 256 or 512 is a
// reasonable choice for a machine dedicated to long prompts. Every microbatch
// produces identical logits and identical state, so this is a throughput and
// scratch-memory choice only, never a semantic one.
struct ExecutionOptions {
    static constexpr std::uint32_t minimum_prefill_ubatch = 1;
    static constexpr std::uint32_t maximum_prefill_ubatch = 4096;

    std::uint32_t prefill_ubatch = 128;
};

// Throws std::invalid_argument when a field is outside its supported range.
void validate_execution_options(const ExecutionOptions &options);

} // namespace adi
