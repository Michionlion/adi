#include "simd_x86.hpp"

#include <cstdint>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#elif (defined(__GNUC__) || defined(__clang__)) && \
    (defined(__x86_64__) || defined(__i386__))
#include <cpuid.h>
#endif

namespace adi::detail {
namespace {

struct Registers {
    std::uint32_t eax = 0;
    std::uint32_t ebx = 0;
    std::uint32_t ecx = 0;
    std::uint32_t edx = 0;
};

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
Registers cpuid(std::uint32_t leaf, std::uint32_t subleaf) noexcept {
    int values[4]{};
    __cpuidex(
        values,
        static_cast<int>(leaf),
        static_cast<int>(subleaf));
    return {
        static_cast<std::uint32_t>(values[0]),
        static_cast<std::uint32_t>(values[1]),
        static_cast<std::uint32_t>(values[2]),
        static_cast<std::uint32_t>(values[3]),
    };
}

std::uint64_t xgetbv(std::uint32_t index) noexcept {
    return _xgetbv(index);
}
#elif (defined(__GNUC__) || defined(__clang__)) && \
    (defined(__x86_64__) || defined(__i386__))
Registers cpuid(std::uint32_t leaf, std::uint32_t subleaf) noexcept {
    Registers result;
    __cpuid_count(
        leaf,
        subleaf,
        result.eax,
        result.ebx,
        result.ecx,
        result.edx);
    return result;
}

std::uint64_t xgetbv(std::uint32_t index) noexcept {
    std::uint32_t eax = 0;
    std::uint32_t edx = 0;
    __asm__ volatile(
        ".byte 0x0f, 0x01, 0xd0"
        : "=a"(eax), "=d"(edx)
        : "c"(index));
    return static_cast<std::uint64_t>(eax) |
           (static_cast<std::uint64_t>(edx) << 32);
}
#endif

constexpr bool bit(std::uint32_t value, std::uint32_t index) noexcept {
    return (value & (1U << index)) != 0;
}

} // namespace

CpuFeatures x86_detect_features() noexcept {
    CpuFeatures features;
#if (defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))) || \
    ((defined(__GNUC__) || defined(__clang__)) && \
     (defined(__x86_64__) || defined(__i386__)))
    const auto maximum = cpuid(0, 0).eax;
    if (maximum < 1) {
        return features;
    }
    const auto leaf1 = cpuid(1, 0);
    const bool osxsave = bit(leaf1.ecx, 27);
    const bool hardware_avx = bit(leaf1.ecx, 28);
    const auto xcr0 = osxsave ? xgetbv(0) : 0;
    const bool avx_state = hardware_avx && (xcr0 & 0x6U) == 0x6U;
    const bool avx512_state =
        avx_state && (xcr0 & 0xE6U) == 0xE6U;
    features.f16c = avx_state && bit(leaf1.ecx, 29);
    if (maximum < 7) {
        return features;
    }

    const auto leaf7 = cpuid(7, 0);
    features.avx2 = avx_state && bit(leaf7.ebx, 5);
    features.avx512f = avx512_state && bit(leaf7.ebx, 16);
    features.avx512bw = features.avx512f && bit(leaf7.ebx, 30);
    features.avx512vbmi = features.avx512f && bit(leaf7.ecx, 1);
    features.avx512vbmi2 = features.avx512f && bit(leaf7.ecx, 6);
    features.avx512vnni = features.avx512f && bit(leaf7.ecx, 11);
    if (leaf7.eax >= 1) {
        const auto leaf71 = cpuid(7, 1);
        features.avx_vnni = avx_state && bit(leaf71.eax, 4);
        features.avx512bf16 = features.avx512f && bit(leaf71.eax, 5);
    }
#endif
    return features;
}

} // namespace adi::detail
