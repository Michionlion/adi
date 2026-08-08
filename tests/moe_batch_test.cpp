// Grouping tokens by expert changes which matmul a token's vector rides in,
// never what that vector is multiplied by or the order the eight expert
// contributions are summed. Routes, weights, and outputs must match the
// unbatched path exactly at every batch size and every routing pattern.
#include "adi/executor.hpp"
#include "adi/model.hpp"
#include "model_test_main.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <span>
#include <utility>
#include <vector>

namespace {

// Worker threads allocate too, so these are atomic rather than plain.
std::atomic<std::size_t> allocations{0};
std::atomic<bool> counting{false};

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

enum class Pattern {
    // Every token routes to the same eight experts: one expert owns the
    // whole batch, the widest possible grouped matmul.
    identical,
    // Independent vectors spread routes across many experts, leaving most
    // grouped ranges short and some experts unused.
    varied,
    // Repeated vectors interleaved with fresh ones produce duplicate routes
    // in a non-contiguous order.
    duplicated,
};

// Where each buffer's storage lives and how large it is. Equal values across
// two calls mean nothing was reallocated between them.
using BufferIdentity = std::vector<std::pair<const void *, std::size_t>>;

template <typename T>
void note(BufferIdentity &identity, const std::vector<T> &values) {
    identity.emplace_back(values.data(), values.capacity());
}

BufferIdentity buffer_identity(const adi::MoeBatchScratch &scratch) {
    BufferIdentity identity;
    note(identity, scratch.router_logits);
    note(identity, scratch.route_order);
    note(identity, scratch.routes);
    note(identity, scratch.counts);
    note(identity, scratch.offsets);
    note(identity, scratch.cursors);
    note(identity, scratch.active);
    note(identity, scratch.grouped_batch);
    note(identity, scratch.route_to_grouped);
    note(identity, scratch.gathered);
    note(identity, scratch.gate);
    note(identity, scratch.up);
    note(identity, scratch.activated);
    note(identity, scratch.projected);
    return identity;
}

std::vector<float> build_inputs(
    std::uint32_t hidden,
    std::uint32_t batch,
    Pattern pattern) {
    std::uint64_t random = 0xB0A710ULL + static_cast<std::uint64_t>(pattern);
    std::vector<float> inputs(
        static_cast<std::size_t>(batch) * hidden);
    std::vector<float> vector(hidden);
    for (auto &value : vector) {
        value = random_float(random);
    }
    for (std::uint32_t index = 0; index < batch; ++index) {
        float *row = inputs.data() + static_cast<std::size_t>(index) * hidden;
        const bool reuse =
            pattern == Pattern::identical ||
            (pattern == Pattern::duplicated && index % 3 != 0);
        if (reuse) {
            std::copy(vector.begin(), vector.end(), row);
            continue;
        }
        for (std::uint32_t element = 0; element < hidden; ++element) {
            row[element] = random_float(random);
        }
        if (pattern == Pattern::duplicated) {
            std::copy(row, row + hidden, vector.begin());
        }
    }
    return inputs;
}

void check_batch(
    const adi::MachModel &model,
    std::uint32_t layer,
    std::uint32_t batch,
    Pattern pattern) {
    const auto hidden = model.config().hidden;
    const auto inputs = build_inputs(hidden, batch, pattern);

    std::vector<float> reference(inputs.size());
    std::vector<std::array<adi::ExpertRoute, 8>> reference_routes(batch);
    adi::MoeScratch reference_scratch;
    for (std::uint32_t index = 0; index < batch; ++index) {
        reference_routes[index] = adi::moe_forward(
            model,
            layer,
            std::span<const float>(inputs).subspan(
                static_cast<std::size_t>(index) * hidden, hidden),
            std::span<float>(reference).subspan(
                static_cast<std::size_t>(index) * hidden, hidden),
            reference_scratch);
    }

    std::vector<float> outputs(inputs.size());
    adi::DecoderBatchScratch scratch;
    adi::moe_forward_batch(model, layer, inputs, outputs, scratch);

    assert(outputs == reference);
    assert(scratch.moe.routes.size() == batch);
    std::size_t distinct = 0;
    for (std::uint32_t index = 0; index < batch; ++index) {
        for (std::uint32_t route = 0; route < 8; ++route) {
            assert(
                scratch.moe.routes[index][route].expert ==
                reference_routes[index][route].expert);
            assert(
                scratch.moe.routes[index][route].weight ==
                reference_routes[index][route].weight);
        }
    }
    for (const auto count : scratch.moe.counts) {
        distinct += count != 0 ? 1 : 0;
    }

    // A second call at the same shape must reuse every buffer rather than
    // rebuild it. Storage identity is the precise statement of that, and
    // unlike an allocation count it does not depend on the ISA or the number
    // of worker threads.
    const auto before = buffer_identity(scratch.moe);
    allocations.store(0, std::memory_order_relaxed);
    counting.store(true, std::memory_order_relaxed);
    adi::moe_forward_batch(model, layer, inputs, outputs, scratch);
    counting.store(false, std::memory_order_relaxed);
    const auto steady = allocations.load(std::memory_order_relaxed);
    assert(outputs == reference);
    assert(buffer_identity(scratch.moe) == before);

    // The allocation count is reported, not asserted. What remains after the
    // grouping buffers moved into MoeBatchScratch comes from dispatching work
    // and from per-worker buffers inside the expert codec kernels, so it
    // tracks the worker count and the ISA rather than anything this function
    // controls.
    std::printf(
        "batch %u pattern %d: exact, %zu active experts, %zu steady allocations\n",
        batch,
        static_cast<int>(pattern),
        distinct,
        steady);
    std::fflush(stdout);
}

} // namespace

void *operator new(std::size_t size) {
    if (counting.load(std::memory_order_relaxed)) {
        allocations.fetch_add(1, std::memory_order_relaxed);
    }
    if (void *memory = std::malloc(size == 0 ? 1 : size)) {
        return memory;
    }
    throw std::bad_alloc();
}

void operator delete(void *memory) noexcept {
    std::free(memory);
}

void operator delete(void *memory, std::size_t) noexcept {
    std::free(memory);
}

int ADI_MODEL_TEST_MAIN(
    int argc,
    adi::test_detail::CommandCharacter **argv) {
    assert(argc == 2);
    const adi::MachModel model(adi::test_detail::model_path(argv[1]));
    constexpr std::uint32_t layer = 0;
    for (const auto pattern :
         {Pattern::identical, Pattern::varied, Pattern::duplicated}) {
        for (const std::uint32_t batch : {1U, 4U, 8U, 64U, 256U}) {
            check_batch(model, layer, batch, pattern);
        }
    }
}
