// The batch-oriented non-expert kernel vectorizes across independent batch
// items only, so every lane must reproduce the scalar kernel's accumulation
// bit for bit, at every batch size including the awkward tails.
#include "adi/kernels.hpp"
#include "ne_batch.hpp"
#include "simd.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <span>
#include <vector>

namespace {

std::uint64_t next_random(std::uint64_t &state) {
    state += 0x9E3779B97F4A7C15ULL;
    std::uint64_t value = state;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

float random_float(std::uint64_t &state) {
    const auto centered =
        static_cast<std::int32_t>((next_random(state) >> 40) & 0xFFFFFFU) -
        0x800000;
    return static_cast<float>(centered) / 8388608.0F;
}

struct Matrix {
    std::vector<std::uint16_t> trellis;
    std::vector<std::int8_t> su;
    std::vector<std::int8_t> sv;
    std::vector<float> tlut;
    adi::MachNeMatrix view;
};

Matrix build_matrix(
    std::uint32_t rows,
    std::uint32_t columns,
    std::uint64_t &state) {
    Matrix matrix;
    const auto tiles = static_cast<std::size_t>(rows / 16) * (columns / 16);
    matrix.trellis.resize(tiles * 64);
    for (auto &word : matrix.trellis) {
        word = static_cast<std::uint16_t>(next_random(state) & 0xFFFFU);
    }
    matrix.su.resize(columns);
    for (auto &sign : matrix.su) {
        sign = (next_random(state) & 1U) != 0 ? 1 : -1;
    }
    matrix.sv.resize(rows);
    for (auto &sign : matrix.sv) {
        sign = (next_random(state) & 1U) != 0 ? 1 : -1;
    }
    matrix.tlut.resize(512 * 2);
    for (auto &value : matrix.tlut) {
        value = random_float(state);
    }
    matrix.view = adi::MachNeMatrix{
        rows,
        columns,
        matrix.trellis,
        matrix.su,
        matrix.sv,
        0.0234375F,
        matrix.tlut,
    };
    return matrix;
}

void check_shape(
    std::uint32_t rows,
    std::uint32_t columns,
    adi::detail::CpuIsa isa,
    std::uint64_t &state) {
    const auto matrix = build_matrix(rows, columns, state);
    for (const std::uint32_t batch :
         {1U, 2U, 3U, 4U, 5U, 7U, 8U, 15U, 16U, 17U, 33U, 64U, 127U}) {
        std::vector<float> inputs(
            static_cast<std::size_t>(batch) * columns);
        for (auto &value : inputs) {
            value = random_float(state);
        }

        std::vector<float> reference(
            static_cast<std::size_t>(batch) * rows);
        adi::ExpertScratch reference_scratch;
        adi::detail::force_cpu_isa_for_testing(adi::detail::CpuIsa::scalar);
        adi::mach_ne_matmul(
            matrix.view, inputs, batch, reference, reference_scratch);

        std::vector<float> actual(reference.size());
        adi::ExpertScratch scratch;
        adi::detail::force_cpu_isa_for_testing(isa);
        adi::mach_ne_matmul(matrix.view, inputs, batch, actual, scratch);
        assert(actual == reference);

        // A batch row must also equal the single-vector kernel on the same
        // ISA exactly: the batch dimension is the only thing that changed.
        std::vector<float> single(rows);
        adi::ExpertScratch single_scratch;
        for (std::uint32_t index = 0; index < batch; ++index) {
            adi::mach_ne_matvec(
                matrix.view,
                std::span<const float>(inputs).subspan(
                    static_cast<std::size_t>(index) * columns, columns),
                single,
                single_scratch);
            for (std::uint32_t row = 0; row < rows; ++row) {
                assert(
                    single[row] ==
                    actual[static_cast<std::size_t>(index) * rows + row]);
            }
        }
        adi::detail::clear_cpu_isa_for_testing();
    }
    const auto name = adi::detail::cpu_isa_name(isa);
    std::printf(
        "%ux%u under %.*s: exact at every batch\n",
        rows,
        columns,
        static_cast<int>(name.size()),
        name.data());
}

} // namespace

int main() {
    const auto native = adi::detail::selected_cpu_isa();
    std::uint64_t state = 0xFEEDBEEFULL;

    // Non-square shapes catch a confused row and column stride, and 16x16
    // is the single-tile case where a mistake in the tail handling shows up
    // immediately.
    for (const auto isa :
         {adi::detail::CpuIsa::avx2,
          adi::detail::CpuIsa::avx512,
          native}) {
        check_shape(16, 16, isa, state);
        check_shape(64, 32, isa, state);
        check_shape(32, 128, isa, state);
    }

    // Lanes must be zero on an ISA without a batch kernel, so the scalar
    // reference path is what runs there.
    adi::detail::force_cpu_isa_for_testing(adi::detail::CpuIsa::scalar);
    assert(adi::detail::ne_batch_lanes() == 0);
    adi::detail::clear_cpu_isa_for_testing();
}
