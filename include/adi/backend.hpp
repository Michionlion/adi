#pragma once

#include "adi/kernels.hpp"

#include <string_view>

namespace adi {

struct Backend {
    std::string_view name;

    decltype(&mach_expert_matvec) expert_matvec;
    decltype(&mach_ne_matvec) ne_matvec;
    decltype(&mach_ne_matmul) ne_matmul;
    decltype(&mach_embedding_row) embedding_row;
    decltype(&mach_head_matvec) head_matvec;
    decltype(&mach_head_matmul) head_matmul;
    decltype(&bf16_matvec) dense_bf16_matvec;
    decltype(&rms_norm) normalize_rms;
};

[[nodiscard]] const Backend &cpu_backend() noexcept;

} // namespace adi
