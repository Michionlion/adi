#include "adi/backend.hpp"
#include "simd.hpp"

namespace adi {

const Backend &cpu_backend() noexcept {
    static const Backend backend{
        detail::cpu_isa_name(detail::selected_cpu_isa()),
        mach_expert_matvec,
        mach_expert_matmul,
        mach_ne_matvec,
        mach_ne_matmul,
        mach_embedding_row,
        mach_head_matvec,
        mach_head_matmul,
        bf16_matvec,
        bf16_matmul,
        rms_norm,
    };
    return backend;
}

} // namespace adi
