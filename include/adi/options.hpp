#pragma once

#include <cstdint>

namespace adi {

// Execution tuning that never changes model semantics. The prefill microbatch
// is the number of prompt tokens that share one packed-weight decode pass. It
// is deliberately not part of the request schema: two requests with the same
// prompt must produce the same tokens regardless of how the server was
// launched.
//
// The default is measured, not assumed. On eight EPYC 9645 cores, a 256-token
// prompt runs at 1.99, 3.35, 4.40, 5.06, 4.25, 3.05, 2.67, and 2.45 tokens
// per second at microbatches of 2, 4, 8, 16, 32, 64, 128, and 256, and a
// 1024-token prompt keeps the same ordering. Sixteen is both the fastest and
// the cheapest of these, at 8 MB of scratch against 32 MB at sixty-four.
//
// The falloff above sixteen is a property of the expert codec, not of
// batching: once several tokens route to the same expert, mach_expert_matmul
// leaves the single-vector path for a scalar batch loop whose row
// accumulators and strided inputs no longer fit L1. Give that kernel the
// batch-lane treatment the non-expert kernel already has and this optimum
// should move up; re-measure before changing it.
struct ExecutionOptions {
    static constexpr std::uint32_t minimum_prefill_ubatch = 1;
    static constexpr std::uint32_t maximum_prefill_ubatch = 4096;

    std::uint32_t prefill_ubatch = 16;
};

// Throws std::invalid_argument when a field is outside its supported range.
void validate_execution_options(const ExecutionOptions &options);

} // namespace adi
