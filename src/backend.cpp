#include "adi/backend.hpp"

namespace adi {

const Backend &cpu_backend() noexcept {
    static constexpr Backend backend{
        "cpu",
        mach_expert_matvec,
        mach_ne_matvec,
        mach_embedding_row,
        mach_head_matvec,
        bf16_matvec,
        rms_norm,
    };
    return backend;
}

} // namespace adi
