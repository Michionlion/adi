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
// expert dispatch handed each expert a single matmul over all of its routed
// rows: mach_expert_matmul's cost per row grew with the batch, so raising the
// microbatch made the dominant kernel slower faster than it made anything
// else faster.
//
// Splitting expert work into twelve-row chunks removed that penalty and
// inverted the curve. On eight EPYC 9645 cores a 256-token prompt now runs at
// 6.31, 7.70, and 8.40 tokens per second at microbatches of 16, 32, and 64,
// where it previously ran at 5.06, 4.25, and 3.05. Sixty-four costs 31.8 MB of
// prefill scratch against 8.0 MB at sixteen, which is the price of the 33%.
//
// Callers who want a different point on that trade — a smaller cache, a
// different core count — set it with --ubatch. Every microbatch produces
// identical logits and identical state, so this is a throughput and
// scratch-memory choice only.
//
// This is an interim value. mach_expert_matmul is still a scalar batch loop;
// once it decodes each packed weight once across a register of batch lanes the
// curve moves again, and the full sweep including 128 and above belongs after
// that lands.
struct ExecutionOptions {
    static constexpr std::uint32_t minimum_prefill_ubatch = 1;
    static constexpr std::uint32_t maximum_prefill_ubatch = 4096;

    std::uint32_t prefill_ubatch = 64;
};

// Throws std::invalid_argument when a field is outside its supported range.
void validate_execution_options(const ExecutionOptions &options);

} // namespace adi
