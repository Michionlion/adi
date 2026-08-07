#include "adi/executor.hpp"
#include "attention.hpp"
#include "parallel.hpp"
#include "profiling_internal.hpp"
#include "simd.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace adi {
namespace {

void normalize_head(
    std::span<float> values,
    std::span<const std::uint16_t> weights,
    float epsilon) {
    if (values.empty() || values.size() != weights.size()) {
        throw std::invalid_argument("attention norm shape mismatch");
    }
    float squares = 0.0F;
    for (const auto value : values) {
        squares += value * value;
    }
    const float scale =
        1.0F / std::sqrt(squares / static_cast<float>(values.size()) + epsilon);
    for (std::size_t index = 0; index < values.size(); ++index) {
        values[index] *= scale * (1.0F + bf16_to_f32(weights[index]));
    }
}

void prepare_rope(
    const MachModel &model,
    std::uint32_t position,
    FullAttentionScratch &scratch) {
    if (scratch.rope_position == position) {
        return;
    }
    const auto divisors = model.rope_theta_divisors();
    for (std::size_t index = 0; index < divisors.size(); ++index) {
        const float theta =
            static_cast<float>(position) / divisors[index];
        scratch.rope_cosine[index] = std::cos(theta);
        scratch.rope_sine[index] = std::sin(theta);
    }
    scratch.rope_position = position;
}

void apply_rope(
    std::span<float> values,
    const FullAttentionScratch &scratch) {
    constexpr std::uint32_t half = 32;
    for (std::uint32_t index = 0; index < half; ++index) {
        const float cosine = scratch.rope_cosine[index];
        const float sine = scratch.rope_sine[index];
        const float first = values[index];
        const float second = values[index + half];
        values[index] = first * cosine - second * sine;
        values[index + half] = second * cosine + first * sine;
    }
}

void gated_delta_recurrent(
    std::span<float> recurrent,
    std::span<const float> query,
    std::span<const float> key,
    std::span<const float> value,
    std::span<const float> alpha,
    std::span<const float> beta,
    std::span<const std::uint16_t> alpha_bias,
    std::span<const std::uint16_t> a_log,
    std::uint32_t key_heads,
    std::uint32_t value_heads,
    std::uint32_t head_size,
    std::span<float> output) {
    const auto state_size =
        static_cast<std::size_t>(value_heads) * head_size * head_size;
    const auto value_size =
        static_cast<std::size_t>(value_heads) * head_size;
    const auto key_size =
        static_cast<std::size_t>(key_heads) * head_size;
    if (recurrent.size() != state_size || query.size() != key_size ||
        key.size() != key_size || value.size() != value_size ||
        alpha.size() != value_heads || beta.size() != value_heads ||
        alpha_bias.size() != value_heads || a_log.size() != value_heads ||
        output.size() != value_size) {
        throw std::invalid_argument("Gated DeltaNet recurrent shape mismatch");
    }

    const float output_scale =
        1.0F / std::sqrt(static_cast<float>(head_size));
    detail::parallel_ranges(
        value_heads,
        1,
        [&](std::uint32_t head_begin, std::uint32_t head_end) {
            for (std::uint32_t head = head_begin;
                 head < head_end;
                 ++head) {
                const auto key_head =
                    gated_delta_key_head(head, value_heads, key_heads);
                const auto q = query.subspan(
                    static_cast<std::size_t>(key_head) * head_size,
                    head_size);
                const auto k = key.subspan(
                    static_cast<std::size_t>(key_head) * head_size,
                    head_size);
                const auto v = value.subspan(
                    static_cast<std::size_t>(head) * head_size,
                    head_size);
                const float beta_value = sigmoid(beta[head]);
                const float biased_alpha =
                    alpha[head] + bf16_to_f32(alpha_bias[head]);
                const float softplus =
                    biased_alpha > 20.0F
                        ? biased_alpha
                        : std::log1p(std::exp(biased_alpha));
                const float decay = std::exp(
                    -std::exp(bf16_to_f32(a_log[head])) * softplus);
                for (std::uint32_t row = 0; row < head_size; ++row) {
                    auto state_row = recurrent.subspan(
                        (static_cast<std::size_t>(head) * head_size + row) *
                            head_size,
                        head_size);
                    output[static_cast<std::size_t>(head) * head_size + row] =
                        detail::gated_delta_update(
                            state_row,
                            q,
                            k,
                            v[row],
                            beta_value,
                            decay) *
                        output_scale;
                }
            }
        });
}

} // namespace

std::array<ExpertRoute, 8> top_experts(std::span<const float> logits) {
    if (logits.size() < 8) {
        throw std::invalid_argument("MoE routing requires at least eight experts");
    }
    std::vector<std::uint32_t> indexes(logits.size());
    std::iota(indexes.begin(), indexes.end(), 0);
    std::partial_sort(
        indexes.begin(),
        indexes.begin() + 8,
        indexes.end(),
        [&](std::uint32_t left, std::uint32_t right) {
            if (logits[left] == logits[right]) {
                return left < right;
            }
            return logits[left] > logits[right];
        });

    const float maximum = logits[indexes[0]];
    std::array<float, 8> selected;
    float selected_sum = 0.0F;
    for (std::size_t index = 0; index < 8; ++index) {
        selected[index] = std::exp(logits[indexes[index]] - maximum);
        selected_sum += selected[index];
    }
    selected_sum = std::max(selected_sum, 6.103515625e-5F);
    std::array<ExpertRoute, 8> routes;
    for (std::size_t index = 0; index < routes.size(); ++index) {
        routes[index] = {
            indexes[index],
            selected[index] / selected_sum,
        };
    }
    return routes;
}

std::uint32_t gated_delta_key_head(
    std::uint32_t value_head,
    std::uint32_t value_heads,
    std::uint32_t key_heads) {
    if (key_heads == 0 || value_heads == 0 || value_heads % key_heads != 0 ||
        value_head >= value_heads) {
        throw std::invalid_argument("Gated DeltaNet head geometry is invalid");
    }
    return value_head / (value_heads / key_heads);
}

std::array<ExpertRoute, 8> moe_forward(
    const MachModel &model,
    std::uint32_t layer,
    std::span<const float> input,
    std::span<float> output,
    MoeScratch &scratch,
    const Backend &backend) {
    detail::KernelTimer timer(KernelKind::moe, input.size());
    const auto &config = model.config();
    if (input.size() != config.hidden || output.size() != config.hidden) {
        throw std::invalid_argument("MoE input shape mismatch");
    }
    const auto &descriptor = model.layer(layer).moe;
    scratch.router_logits.resize(config.experts);
    backend.dense_bf16_matvec(
        descriptor.router, input, scratch.router_logits);
    const auto routes = top_experts(scratch.router_logits);

    std::fill(output.begin(), output.end(), 0.0F);
    detail::parallel_tasks(
        static_cast<std::uint32_t>(routes.size()),
        [&](std::uint32_t route_index) {
            const auto route = routes[route_index];
            auto &worker = scratch.experts[route_index];
            worker.gate.resize(config.expert_hidden);
            worker.up.resize(config.expert_hidden);
            worker.activated.resize(config.expert_hidden);
            worker.projected.resize(config.hidden);
            backend.expert_matvec(
                descriptor.experts[route.expert][0],
                input,
                worker.gate,
                worker.codec);
            backend.expert_matvec(
                descriptor.experts[route.expert][1],
                input,
                worker.up,
                worker.codec);
            for (std::size_t index = 0; index < worker.activated.size(); ++index) {
                worker.activated[index] =
                    silu(worker.gate[index]) * worker.up[index];
            }
            backend.expert_matvec(
                descriptor.experts[route.expert][2],
                worker.activated,
                worker.projected,
                worker.codec);
        });
    for (std::size_t route_index = 0; route_index < routes.size(); ++route_index) {
        const auto &projected = scratch.experts[route_index].projected;
        for (std::size_t index = 0; index < output.size(); ++index) {
            output[index] += routes[route_index].weight * projected[index];
        }
    }

    scratch.shared_gate.resize(config.expert_hidden);
    scratch.shared_up.resize(config.expert_hidden);
    scratch.shared_down.resize(config.hidden);
    backend.ne_matvec(
        descriptor.shared_gate,
        input,
        scratch.shared_gate,
        scratch.shared_codec);
    backend.ne_matvec(
        descriptor.shared_up,
        input,
        scratch.shared_up,
        scratch.shared_codec);
    for (std::size_t index = 0; index < scratch.shared_gate.size(); ++index) {
        scratch.shared_gate[index] =
            silu(scratch.shared_gate[index]) * scratch.shared_up[index];
    }
    backend.ne_matvec(
        descriptor.shared_down,
        scratch.shared_gate,
        scratch.shared_down,
        scratch.shared_codec);

    float shared_weight = 0.0F;
    backend.dense_bf16_matvec(
        descriptor.shared_expert_gate,
        input,
        std::span<float>(&shared_weight, 1));
    shared_weight = sigmoid(shared_weight);
    for (std::size_t index = 0; index < output.size(); ++index) {
        output[index] += shared_weight * scratch.shared_down[index];
    }
    return routes;
}

void full_attention_forward(
    const MachModel &model,
    std::uint32_t layer,
    std::uint32_t position,
    std::span<const float> input,
    std::span<float> output,
    FullAttentionState &state,
    FullAttentionScratch &scratch,
    const Backend &backend) {
    detail::KernelTimer timer(KernelKind::full_attention, position + 1);
    constexpr std::uint32_t query_heads = 16;
    constexpr std::uint32_t kv_heads = 2;
    constexpr std::uint32_t head_size = 256;
    constexpr std::uint32_t query_stride = 2 * head_size;
    constexpr float epsilon = 1.0e-6F;
    const auto &config = model.config();
    if (input.size() != config.hidden || output.size() != config.hidden ||
        state.keys.size() != state.values.size() ||
        state.keys.size() != static_cast<std::size_t>(position) * kv_heads * head_size) {
        throw std::invalid_argument("full attention state or input shape mismatch");
    }
    const auto &descriptor = model.layer(layer).full;
    scratch.query_gate.resize(query_heads * query_stride);
    scratch.key.resize(kv_heads * head_size);
    scratch.value.resize(kv_heads * head_size);
    scratch.attended.resize(query_heads * head_size);
    backend.ne_matvec(
        descriptor.query,
        input,
        scratch.query_gate,
        scratch.codec);
    backend.ne_matvec(
        descriptor.key,
        input,
        scratch.key,
        scratch.codec);
    backend.ne_matvec(
        descriptor.value,
        input,
        scratch.value,
        scratch.codec);

    prepare_rope(model, position, scratch);
    for (std::uint32_t head = 0; head < query_heads; ++head) {
        auto query = std::span<float>(
            scratch.query_gate.data() + static_cast<std::size_t>(head) * query_stride,
            head_size);
        normalize_head(query, descriptor.query_norm, epsilon);
        apply_rope(query, scratch);
    }
    for (std::uint32_t head = 0; head < kv_heads; ++head) {
        auto key = std::span<float>(
            scratch.key.data() + static_cast<std::size_t>(head) * head_size,
            head_size);
        normalize_head(key, descriptor.key_norm, epsilon);
        apply_rope(key, scratch);
    }
    state.keys.insert(state.keys.end(), scratch.key.begin(), scratch.key.end());
    state.values.insert(state.values.end(), scratch.value.begin(), scratch.value.end());

    scratch.contiguous_queries.resize(
        static_cast<std::size_t>(query_heads) * head_size);
    for (std::uint32_t query_head = 0;
         query_head < query_heads;
         ++query_head) {
        std::copy_n(
            scratch.query_gate.begin() +
                static_cast<std::size_t>(query_head) * query_stride,
            head_size,
            scratch.contiguous_queries.begin() +
                static_cast<std::size_t>(query_head) * head_size);
    }
    detail::grouped_query_online_attention(
        scratch.contiguous_queries,
        state.keys,
        state.values,
        query_heads,
        kv_heads,
        head_size,
        scratch.attended);
    for (std::uint32_t query_head = 0; query_head < query_heads; ++query_head) {
        auto attended = std::span<float>(
            scratch.attended.data() +
                static_cast<std::size_t>(query_head) * head_size,
            head_size);
        const auto gate_offset =
            static_cast<std::size_t>(query_head) * query_stride + head_size;
        for (std::uint32_t index = 0; index < head_size; ++index) {
            attended[index] *= sigmoid(scratch.query_gate[gate_offset + index]);
        }
    }
    backend.ne_matvec(
        descriptor.output,
        scratch.attended,
        output,
        scratch.codec);
}

void linear_attention_forward(
    const MachModel &model,
    std::uint32_t layer,
    std::span<const float> input,
    std::span<float> output,
    LinearAttentionState &state,
    LinearAttentionScratch &scratch,
    const Backend &backend) {
    detail::KernelTimer timer(KernelKind::linear_attention, input.size());
    constexpr std::uint32_t key_heads = 16;
    constexpr std::uint32_t value_heads = 32;
    constexpr std::uint32_t head_size = 128;
    constexpr std::uint32_t qk_size = key_heads * head_size;
    constexpr std::uint32_t value_size = value_heads * head_size;
    constexpr std::uint32_t channels = 2 * qk_size + value_size;
    constexpr std::uint32_t convolution_width = 4;
    constexpr float epsilon = 1.0e-6F;
    if (input.size() != model.config().hidden ||
        output.size() != model.config().hidden) {
        throw std::invalid_argument("linear attention input shape mismatch");
    }
    const auto &descriptor = model.layer(layer).linear;
    scratch.qkv.resize(channels);
    scratch.gate.resize(value_size);
    scratch.convolved.resize(channels);
    scratch.alpha.resize(value_heads);
    scratch.beta.resize(value_heads);
    scratch.recurrent_output.resize(value_size);
    scratch.normalized.resize(value_size);
    backend.ne_matvec(
        descriptor.qkv,
        input,
        scratch.qkv,
        scratch.codec);
    backend.ne_matvec(
        descriptor.gate,
        input,
        scratch.gate,
        scratch.codec);
    backend.dense_bf16_matvec(
        descriptor.alpha,
        input,
        scratch.alpha);
    backend.dense_bf16_matvec(
        descriptor.beta,
        input,
        scratch.beta);

    const auto convolution_weights = descriptor.convolution;
    if (convolution_weights.size() !=
        static_cast<std::size_t>(channels) * convolution_width) {
        throw std::runtime_error("linear attention convolution shape mismatch");
    }
    if (state.convolution.empty()) {
        state.convolution.assign(
            static_cast<std::size_t>(channels) * (convolution_width - 1), 0.0F);
    }
    if (state.convolution.size() !=
        static_cast<std::size_t>(channels) * (convolution_width - 1)) {
        throw std::invalid_argument("linear attention convolution state mismatch");
    }
    for (std::uint32_t channel = 0; channel < channels; ++channel) {
        const auto history = static_cast<std::size_t>(channel) * 3;
        const auto weights = static_cast<std::size_t>(channel) * convolution_width;
        float value = 0.0F;
        for (std::uint32_t index = 0; index < 3; ++index) {
            value += state.convolution[history + index] *
                     bf16_to_f32(convolution_weights[weights + index]);
        }
        value += scratch.qkv[channel] *
                 bf16_to_f32(convolution_weights[weights + 3]);
        scratch.convolved[channel] = silu(value);
        state.convolution[history] = state.convolution[history + 1];
        state.convolution[history + 1] = state.convolution[history + 2];
        state.convolution[history + 2] = scratch.qkv[channel];
    }

    auto query = std::span<float>(scratch.convolved.data(), qk_size);
    auto key = std::span<float>(scratch.convolved.data() + qk_size, qk_size);
    auto value =
        std::span<float>(scratch.convolved.data() + 2 * qk_size, value_size);
    for (std::uint32_t head = 0; head < key_heads; ++head) {
        for (auto vector : {
                 query.subspan(static_cast<std::size_t>(head) * head_size, head_size),
                 key.subspan(static_cast<std::size_t>(head) * head_size, head_size)}) {
            l2_normalize(vector, epsilon);
        }
    }

    const auto alpha_bias = descriptor.alpha_bias;
    const auto a_log = descriptor.a_log;
    if (alpha_bias.size() != value_heads || a_log.size() != value_heads) {
        throw std::runtime_error("linear attention decay shape mismatch");
    }
    if (state.recurrent.empty()) {
        state.recurrent.assign(
            static_cast<std::size_t>(value_heads) * head_size * head_size, 0.0F);
    }
    if (state.recurrent.size() !=
        static_cast<std::size_t>(value_heads) * head_size * head_size) {
        throw std::invalid_argument("linear attention recurrent state mismatch");
    }

    gated_delta_recurrent(
        state.recurrent,
        query,
        key,
        value,
        scratch.alpha,
        scratch.beta,
        alpha_bias,
        a_log,
        key_heads,
        value_heads,
        head_size,
        scratch.recurrent_output);

    const auto norm = descriptor.norm;
    for (std::uint32_t head = 0; head < value_heads; ++head) {
        const auto source = std::span<const float>(
            scratch.recurrent_output.data() +
                static_cast<std::size_t>(head) * head_size,
            head_size);
        auto destination = std::span<float>(
            scratch.normalized.data() + static_cast<std::size_t>(head) * head_size,
            head_size);
        backend.normalize_rms(source, norm, 0.0F, epsilon, destination);
        for (std::uint32_t index = 0; index < head_size; ++index) {
            destination[index] *= silu(
                scratch.gate[static_cast<std::size_t>(head) * head_size + index]);
        }
    }
    backend.ne_matvec(
        descriptor.output,
        scratch.normalized,
        output,
        scratch.codec);
}

namespace {

void full_attention_forward_batch(
    const MachModel &model,
    std::uint32_t layer,
    std::span<const std::uint32_t> positions,
    std::span<const float> inputs,
    std::span<FullAttentionState *> states,
    std::span<FullAttentionScratch *> scratches,
    std::span<float> outputs,
    DecoderBatchScratch &batch_scratch,
    std::span<const float> rope_cosine,
    std::span<const float> rope_sine,
    const Backend &backend) {
    constexpr std::uint32_t query_heads = 16;
    constexpr std::uint32_t kv_heads = 2;
    constexpr std::uint32_t head_size = 256;
    constexpr std::uint32_t query_stride = 2 * head_size;
    constexpr float epsilon = 1.0e-6F;
    const auto batch = positions.size();
    const auto &descriptor = model.layer(layer).full;
    if (batch == 0 || batch > std::numeric_limits<std::uint32_t>::max() ||
        states.size() != batch || scratches.size() != batch ||
        inputs.size() != batch * model.config().hidden ||
        outputs.size() != batch * model.config().hidden) {
        throw std::invalid_argument("full attention batch shape mismatch");
    }
    if ((!rope_cosine.empty() || !rope_sine.empty()) &&
        (rope_cosine.size() != batch * 32 ||
         rope_sine.size() != batch * 32)) {
        throw std::invalid_argument("full attention RoPE batch shape mismatch");
    }
    detail::KernelTimer timer(KernelKind::full_attention, batch);
    batch_scratch.projection_0.resize(batch * query_heads * query_stride);
    batch_scratch.projection_1.resize(batch * kv_heads * head_size);
    batch_scratch.projection_2.resize(batch * kv_heads * head_size);
    batch_scratch.projection_3.resize(batch * query_heads * head_size);
    backend.ne_matmul(
        descriptor.query,
        inputs,
        static_cast<std::uint32_t>(batch),
        batch_scratch.projection_0,
        batch_scratch.codec);
    backend.ne_matmul(
        descriptor.key,
        inputs,
        static_cast<std::uint32_t>(batch),
        batch_scratch.projection_1,
        batch_scratch.codec);
    backend.ne_matmul(
        descriptor.value,
        inputs,
        static_cast<std::uint32_t>(batch),
        batch_scratch.projection_2,
        batch_scratch.codec);

    for (std::size_t batch_index = 0;
         batch_index < batch;
         ++batch_index) {
        auto &state = *states[batch_index];
        auto &scratch = *scratches[batch_index];
        const auto position = positions[batch_index];
        if (state.keys.size() != state.values.size() ||
            state.keys.size() !=
                static_cast<std::size_t>(position) * kv_heads * head_size) {
            throw std::invalid_argument("full attention batch state mismatch");
        }
        scratch.query_gate.assign(
            batch_scratch.projection_0.begin() +
                batch_index * query_heads * query_stride,
            batch_scratch.projection_0.begin() +
                (batch_index + 1) * query_heads * query_stride);
        scratch.key.assign(
            batch_scratch.projection_1.begin() +
                batch_index * kv_heads * head_size,
            batch_scratch.projection_1.begin() +
                (batch_index + 1) * kv_heads * head_size);
        scratch.value.assign(
            batch_scratch.projection_2.begin() +
                batch_index * kv_heads * head_size,
            batch_scratch.projection_2.begin() +
                (batch_index + 1) * kv_heads * head_size);
        scratch.attended.resize(query_heads * head_size);
        if (!rope_cosine.empty()) {
            std::copy_n(
                rope_cosine.begin() + batch_index * 32,
                32,
                scratch.rope_cosine.begin());
            std::copy_n(
                rope_sine.begin() + batch_index * 32,
                32,
                scratch.rope_sine.begin());
            scratch.rope_position = position;
        }
        prepare_rope(model, position, scratch);
        for (std::uint32_t head = 0; head < query_heads; ++head) {
            auto query = std::span<float>(
                scratch.query_gate.data() +
                    static_cast<std::size_t>(head) * query_stride,
                head_size);
            normalize_head(query, descriptor.query_norm, epsilon);
            apply_rope(query, scratch);
        }
        for (std::uint32_t head = 0; head < kv_heads; ++head) {
            auto key = std::span<float>(
                scratch.key.data() +
                    static_cast<std::size_t>(head) * head_size,
                head_size);
            normalize_head(key, descriptor.key_norm, epsilon);
            apply_rope(key, scratch);
        }
        state.keys.insert(
            state.keys.end(),
            scratch.key.begin(),
            scratch.key.end());
        state.values.insert(
            state.values.end(),
            scratch.value.begin(),
            scratch.value.end());
        scratch.contiguous_queries.resize(query_heads * head_size);
        for (std::uint32_t query_head = 0;
             query_head < query_heads;
             ++query_head) {
            std::copy_n(
                scratch.query_gate.begin() +
                    static_cast<std::size_t>(query_head) * query_stride,
                head_size,
                scratch.contiguous_queries.begin() +
                    static_cast<std::size_t>(query_head) * head_size);
        }
        detail::grouped_query_online_attention(
            scratch.contiguous_queries,
            state.keys,
            state.values,
            query_heads,
            kv_heads,
            head_size,
            scratch.attended);
        for (std::uint32_t query_head = 0;
             query_head < query_heads;
             ++query_head) {
            const auto gate_offset =
                static_cast<std::size_t>(query_head) * query_stride +
                head_size;
            for (std::uint32_t index = 0; index < head_size; ++index) {
                scratch.attended[
                    static_cast<std::size_t>(query_head) * head_size +
                    index] *= sigmoid(scratch.query_gate[gate_offset + index]);
            }
        }
        std::copy(
            scratch.attended.begin(),
            scratch.attended.end(),
            batch_scratch.projection_3.begin() +
                batch_index * query_heads * head_size);
    }
    backend.ne_matmul(
        descriptor.output,
        batch_scratch.projection_3,
        static_cast<std::uint32_t>(batch),
        outputs,
        batch_scratch.codec);
}

void linear_attention_forward_batch(
    const MachModel &model,
    std::uint32_t layer,
    std::span<const float> inputs,
    std::span<LinearAttentionState *> states,
    std::span<LinearAttentionScratch *> scratches,
    std::span<float> outputs,
    DecoderBatchScratch &batch_scratch,
    const Backend &backend) {
    constexpr std::uint32_t key_heads = 16;
    constexpr std::uint32_t value_heads = 32;
    constexpr std::uint32_t head_size = 128;
    constexpr std::uint32_t qk_size = key_heads * head_size;
    constexpr std::uint32_t value_size = value_heads * head_size;
    constexpr std::uint32_t channels = 2 * qk_size + value_size;
    constexpr std::uint32_t convolution_width = 4;
    constexpr float epsilon = 1.0e-6F;
    const auto batch = states.size();
    const auto &descriptor = model.layer(layer).linear;
    if (batch == 0 || batch > std::numeric_limits<std::uint32_t>::max() ||
        scratches.size() != batch ||
        inputs.size() != batch * model.config().hidden ||
        outputs.size() != batch * model.config().hidden) {
        throw std::invalid_argument("linear attention batch shape mismatch");
    }
    detail::KernelTimer timer(KernelKind::linear_attention, batch);
    batch_scratch.projection_0.resize(batch * channels);
    batch_scratch.projection_1.resize(batch * value_size);
    batch_scratch.projection_2.resize(batch * value_heads);
    batch_scratch.projection_3.resize(batch * value_heads);
    batch_scratch.projection_4.resize(batch * value_size);
    backend.ne_matmul(
        descriptor.qkv,
        inputs,
        static_cast<std::uint32_t>(batch),
        batch_scratch.projection_0,
        batch_scratch.codec);
    backend.ne_matmul(
        descriptor.gate,
        inputs,
        static_cast<std::uint32_t>(batch),
        batch_scratch.projection_1,
        batch_scratch.codec);
    backend.dense_bf16_matmul(
        descriptor.alpha,
        inputs,
        static_cast<std::uint32_t>(batch),
        batch_scratch.projection_2);
    backend.dense_bf16_matmul(
        descriptor.beta,
        inputs,
        static_cast<std::uint32_t>(batch),
        batch_scratch.projection_3);

    for (std::size_t batch_index = 0;
         batch_index < batch;
         ++batch_index) {
        auto &state = *states[batch_index];
        auto &scratch = *scratches[batch_index];
        scratch.qkv.assign(
            batch_scratch.projection_0.begin() + batch_index * channels,
            batch_scratch.projection_0.begin() + (batch_index + 1) * channels);
        scratch.gate.assign(
            batch_scratch.projection_1.begin() + batch_index * value_size,
            batch_scratch.projection_1.begin() + (batch_index + 1) * value_size);
        scratch.alpha.assign(
            batch_scratch.projection_2.begin() + batch_index * value_heads,
            batch_scratch.projection_2.begin() + (batch_index + 1) * value_heads);
        scratch.beta.assign(
            batch_scratch.projection_3.begin() + batch_index * value_heads,
            batch_scratch.projection_3.begin() + (batch_index + 1) * value_heads);
        scratch.convolved.resize(channels);
        scratch.recurrent_output.resize(value_size);
        scratch.normalized.resize(value_size);

        if (descriptor.convolution.size() !=
            static_cast<std::size_t>(channels) * convolution_width) {
            throw std::runtime_error(
                "linear attention convolution shape mismatch");
        }
        if (state.convolution.empty()) {
            state.convolution.assign(
                static_cast<std::size_t>(channels) *
                    (convolution_width - 1),
                0.0F);
        }
        if (state.convolution.size() !=
            static_cast<std::size_t>(channels) *
                (convolution_width - 1)) {
            throw std::invalid_argument(
                "linear attention convolution state mismatch");
        }
        for (std::uint32_t channel = 0;
             channel < channels;
             ++channel) {
            const auto history = static_cast<std::size_t>(channel) * 3;
            const auto weights =
                static_cast<std::size_t>(channel) * convolution_width;
            float value = 0.0F;
            for (std::uint32_t index = 0; index < 3; ++index) {
                value +=
                    state.convolution[history + index] *
                    bf16_to_f32(descriptor.convolution[weights + index]);
            }
            value +=
                scratch.qkv[channel] *
                bf16_to_f32(descriptor.convolution[weights + 3]);
            scratch.convolved[channel] = silu(value);
            state.convolution[history] =
                state.convolution[history + 1];
            state.convolution[history + 1] =
                state.convolution[history + 2];
            state.convolution[history + 2] = scratch.qkv[channel];
        }

        auto query = std::span<float>(scratch.convolved.data(), qk_size);
        auto key = std::span<float>(
            scratch.convolved.data() + qk_size,
            qk_size);
        auto value = std::span<float>(
            scratch.convolved.data() + 2 * qk_size,
            value_size);
        for (std::uint32_t head = 0; head < key_heads; ++head) {
            for (auto vector : {
                     query.subspan(
                         static_cast<std::size_t>(head) * head_size,
                         head_size),
                     key.subspan(
                         static_cast<std::size_t>(head) * head_size,
                         head_size),
                }) {
                l2_normalize(vector, epsilon);
            }
        }
        if (descriptor.alpha_bias.size() != value_heads ||
            descriptor.a_log.size() != value_heads) {
            throw std::runtime_error(
                "linear attention decay shape mismatch");
        }
        if (state.recurrent.empty()) {
            state.recurrent.assign(
                static_cast<std::size_t>(value_heads) * head_size *
                    head_size,
                0.0F);
        }
        if (state.recurrent.size() !=
            static_cast<std::size_t>(value_heads) * head_size *
                head_size) {
            throw std::invalid_argument(
                "linear attention recurrent state mismatch");
        }
        gated_delta_recurrent(
            state.recurrent,
            query,
            key,
            value,
            scratch.alpha,
            scratch.beta,
            descriptor.alpha_bias,
            descriptor.a_log,
            key_heads,
            value_heads,
            head_size,
            scratch.recurrent_output);

        for (std::uint32_t head = 0; head < value_heads; ++head) {
            const auto source = std::span<const float>(
                scratch.recurrent_output.data() +
                    static_cast<std::size_t>(head) * head_size,
                head_size);
            auto destination = std::span<float>(
                scratch.normalized.data() +
                    static_cast<std::size_t>(head) * head_size,
                head_size);
            backend.normalize_rms(
                source,
                descriptor.norm,
                0.0F,
                epsilon,
                destination);
            for (std::uint32_t index = 0; index < head_size; ++index) {
                destination[index] *= silu(
                    scratch.gate[
                        static_cast<std::size_t>(head) * head_size +
                        index]);
            }
        }
        std::copy(
            scratch.normalized.begin(),
            scratch.normalized.end(),
            batch_scratch.projection_4.begin() +
                batch_index * value_size);
    }
    backend.ne_matmul(
        descriptor.output,
        batch_scratch.projection_4,
        static_cast<std::uint32_t>(batch),
        outputs,
        batch_scratch.codec);
}

void moe_forward_batch(
    const MachModel &model,
    std::uint32_t layer,
    std::span<const float> inputs,
    std::span<float> outputs,
    DecoderBatchScratch &batch_scratch,
    const Backend &backend) {
    struct RouteReference {
        std::uint32_t batch;
        std::uint32_t route;
    };

    const auto &config = model.config();
    const auto batch = inputs.size() / config.hidden;
    const auto &descriptor = model.layer(layer).moe;
    if (batch == 0 || batch > std::numeric_limits<std::uint32_t>::max() ||
        inputs.size() != batch * config.hidden ||
        outputs.size() != batch * config.hidden) {
        throw std::invalid_argument("MoE batch shape mismatch");
    }
    detail::KernelTimer timer(KernelKind::moe, batch * config.hidden);
    batch_scratch.projection_0.resize(batch * config.experts);
    backend.dense_bf16_matmul(
        descriptor.router,
        inputs,
        static_cast<std::uint32_t>(batch),
        batch_scratch.projection_0);

    std::vector<std::array<ExpertRoute, 8>> routes(batch);
    std::vector<std::vector<RouteReference>> grouped(config.experts);
    for (std::uint32_t batch_index = 0;
         batch_index < batch;
         ++batch_index) {
        routes[batch_index] = top_experts(
            std::span<const float>(batch_scratch.projection_0)
                .subspan(
                    static_cast<std::size_t>(batch_index) *
                        config.experts,
                    config.experts));
        for (std::uint32_t route = 0; route < 8; ++route) {
            grouped[routes[batch_index][route].expert].push_back(
                {batch_index, route});
        }
    }
    batch_scratch.moe_route_outputs.assign(
        batch * 8 * config.hidden,
        0.0F);
    std::vector<std::uint32_t> active_experts;
    for (std::uint32_t expert = 0; expert < config.experts; ++expert) {
        if (!grouped[expert].empty()) {
            active_experts.push_back(expert);
        }
    }
    detail::parallel_tasks(
        static_cast<std::uint32_t>(active_experts.size()),
        [&](std::uint32_t active_index) {
        const auto expert = active_experts[active_index];
        const auto &references = grouped[expert];
        const auto expert_batch =
            static_cast<std::uint32_t>(references.size());
        std::vector<float> gathered(
            static_cast<std::size_t>(expert_batch) * config.hidden);
        for (std::uint32_t index = 0; index < expert_batch; ++index) {
            std::copy_n(
                inputs.begin() +
                    static_cast<std::size_t>(references[index].batch) *
                        config.hidden,
                config.hidden,
                gathered.begin() +
                    static_cast<std::size_t>(index) * config.hidden);
        }
        std::vector<float> gate(
            static_cast<std::size_t>(expert_batch) *
            config.expert_hidden);
        std::vector<float> up(gate.size());
        std::vector<float> activated(gate.size());
        std::vector<float> projected(
            static_cast<std::size_t>(expert_batch) * config.hidden);
        ExpertScratch codec;
        backend.expert_matmul(
            descriptor.experts[expert][0],
            gathered,
            expert_batch,
            gate,
            codec);
        backend.expert_matmul(
            descriptor.experts[expert][1],
            gathered,
            expert_batch,
            up,
            codec);
        for (std::size_t index = 0; index < activated.size(); ++index) {
            activated[index] = silu(gate[index]) * up[index];
        }
        backend.expert_matmul(
            descriptor.experts[expert][2],
            activated,
            expert_batch,
            projected,
            codec);
        for (std::uint32_t index = 0; index < expert_batch; ++index) {
            const auto &reference = references[index];
            std::copy_n(
                projected.begin() +
                    static_cast<std::size_t>(index) * config.hidden,
                config.hidden,
                batch_scratch.moe_route_outputs.begin() +
                    (static_cast<std::size_t>(reference.batch) * 8 +
                     reference.route) *
                        config.hidden);
        }
    });
    std::fill(outputs.begin(), outputs.end(), 0.0F);
    for (std::uint32_t batch_index = 0;
         batch_index < batch;
         ++batch_index) {
        auto destination = outputs.subspan(
            static_cast<std::size_t>(batch_index) * config.hidden,
            config.hidden);
        for (std::uint32_t route = 0; route < 8; ++route) {
            const auto source = std::span<const float>(
                batch_scratch.moe_route_outputs)
                                    .subspan(
                                        (static_cast<std::size_t>(
                                             batch_index) *
                                             8 +
                                         route) *
                                            config.hidden,
                                        config.hidden);
            for (std::uint32_t index = 0;
                 index < config.hidden;
                 ++index) {
                destination[index] +=
                    routes[batch_index][route].weight * source[index];
            }
        }
    }

    batch_scratch.projection_0.resize(
        batch * config.expert_hidden);
    batch_scratch.projection_1.resize(
        batch * config.expert_hidden);
    backend.ne_matmul(
        descriptor.shared_gate,
        inputs,
        static_cast<std::uint32_t>(batch),
        batch_scratch.projection_0,
        batch_scratch.codec);
    backend.ne_matmul(
        descriptor.shared_up,
        inputs,
        static_cast<std::uint32_t>(batch),
        batch_scratch.projection_1,
        batch_scratch.codec);
    for (std::size_t index = 0;
         index < batch_scratch.projection_0.size();
         ++index) {
        batch_scratch.projection_0[index] =
            silu(batch_scratch.projection_0[index]) *
            batch_scratch.projection_1[index];
    }
    batch_scratch.projection_2.resize(batch * config.hidden);
    backend.ne_matmul(
        descriptor.shared_down,
        batch_scratch.projection_0,
        static_cast<std::uint32_t>(batch),
        batch_scratch.projection_2,
        batch_scratch.codec);
    batch_scratch.projection_3.resize(batch);
    backend.dense_bf16_matmul(
        descriptor.shared_expert_gate,
        inputs,
        static_cast<std::uint32_t>(batch),
        batch_scratch.projection_3);
    for (std::uint32_t batch_index = 0;
         batch_index < batch;
         ++batch_index) {
        const float shared_weight =
            sigmoid(batch_scratch.projection_3[batch_index]);
        for (std::uint32_t index = 0; index < config.hidden; ++index) {
            outputs[
                static_cast<std::size_t>(batch_index) * config.hidden +
                index] +=
                shared_weight *
                batch_scratch.projection_2[
                    static_cast<std::size_t>(batch_index) *
                        config.hidden +
                    index];
        }
    }
}

void initialize_hidden(
    const MachModel &model,
    std::uint32_t token,
    DecoderScratch &scratch,
    const Backend &backend) {
    const auto &config = model.config();
    if (token >= config.vocabulary) {
        throw std::invalid_argument("decoder token is out of range");
    }
    scratch.hidden.resize(config.hidden);
    scratch.normalized.resize(config.hidden);
    scratch.attention.resize(config.hidden);
    scratch.feed_forward.resize(config.hidden);
    backend.embedding_row(model.embedding(), token, scratch.hidden);
}

void decode_layer(
    const MachModel &model,
    std::uint32_t layer,
    std::uint32_t position,
    DecoderState &state,
    DecoderScratch &scratch,
    const Backend &backend) {
    const auto &descriptor = model.layer(layer);
    detail::KernelTimer layer_timer(
        KernelKind::decoder_layer,
        model.config().hidden);
    backend.normalize_rms(
        scratch.hidden,
        descriptor.input_norm,
        1.0F,
        1.0e-6F,
        scratch.normalized);
    if (descriptor.full_attention) {
        full_attention_forward(
            model,
            layer,
            position,
            scratch.normalized,
            scratch.attention,
            state.full_attention[layer],
            scratch.full_attention,
            backend);
    } else {
        linear_attention_forward(
            model,
            layer,
            scratch.normalized,
            scratch.attention,
            state.linear_attention[layer],
            scratch.linear_attention,
            backend);
    }
    for (std::size_t index = 0; index < scratch.hidden.size(); ++index) {
        scratch.hidden[index] += scratch.attention[index];
    }
    backend.normalize_rms(
        scratch.hidden,
        descriptor.post_attention_norm,
        1.0F,
        1.0e-6F,
        scratch.normalized);
    (void)moe_forward(
        model,
        layer,
        scratch.normalized,
        scratch.feed_forward,
        scratch.moe,
        backend);
    for (std::size_t index = 0; index < scratch.hidden.size(); ++index) {
        scratch.hidden[index] += scratch.feed_forward[index];
    }
}

void finalize_hidden(
    const MachModel &model,
    DecoderScratch &scratch,
    const Backend &backend) {
    backend.normalize_rms(
        scratch.hidden,
        model.final_norm(),
        1.0F,
        1.0e-6F,
        scratch.normalized);
}

void decode_hidden(
    const MachModel &model,
    std::uint32_t token,
    DecoderState &state,
    DecoderScratch &scratch,
    const Backend &backend) {
    initialize_hidden(model, token, scratch, backend);
    for (std::uint32_t layer = 0; layer < model.config().layers; ++layer) {
        decode_layer(
            model,
            layer,
            state.position,
            state,
            scratch,
            backend);
    }
    finalize_hidden(model, scratch, backend);
    ++state.position;
}

} // namespace

void decode_token(
    const MachModel &model,
    std::uint32_t token,
    DecoderState &state,
    std::span<float> logits,
    DecoderScratch &scratch,
    const Backend &backend) {
    const auto &config = model.config();
    if (!logits.empty() && logits.size() != config.vocabulary) {
        throw std::invalid_argument("decoder logits shape mismatch");
    }
    if (state.position >= config.context) {
        throw std::out_of_range("decoder context is exhausted");
    }
    decode_hidden(model, token, state, scratch, backend);
    if (logits.empty()) {
        return;
    }
    for (std::uint32_t chunk = 0; chunk < 8; ++chunk) {
        const auto head = model.head_chunk(chunk);
        backend.head_matvec(
            head,
            scratch.normalized,
            logits.subspan(static_cast<std::size_t>(chunk) * head.rows, head.rows));
    }
}

void decode_batch(
    const MachModel &model,
    std::span<const std::uint32_t> tokens,
    std::span<DecoderState> states,
    std::span<float> logits,
    std::span<DecoderScratch> scratches,
    DecoderBatchScratch &batch_scratch,
    const Backend &backend) {
    const auto &config = model.config();
    const auto batch = tokens.size();
    if (batch == 0 || batch > std::numeric_limits<std::uint32_t>::max() ||
        states.size() != batch || scratches.size() != batch ||
        logits.size() != batch * config.vocabulary) {
        throw std::invalid_argument("decoder batch shape mismatch");
    }
    if (std::any_of(tokens.begin(), tokens.end(), [&](std::uint32_t token) {
            return token >= config.vocabulary;
        })) {
        throw std::invalid_argument("decoder batch token is out of range");
    }
    if (std::any_of(states.begin(), states.end(), [&](const DecoderState &state) {
            return state.position >= config.context;
        })) {
        throw std::out_of_range("decoder context is exhausted");
    }

    for (std::size_t batch_index = 0; batch_index < batch; ++batch_index) {
        initialize_hidden(
            model,
            tokens[batch_index],
            scratches[batch_index],
            backend);
    }
    std::vector<std::uint32_t> positions(batch);
    std::vector<FullAttentionState *> full_states(batch);
    std::vector<FullAttentionScratch *> full_scratches(batch);
    std::vector<LinearAttentionState *> linear_states(batch);
    std::vector<LinearAttentionScratch *> linear_scratches(batch);
    for (std::uint32_t layer = 0; layer < config.layers; ++layer) {
        const auto &descriptor = model.layer(layer);
        if (descriptor.full_attention) {
            batch_scratch.projection_5.resize(batch * config.hidden);
            for (std::size_t batch_index = 0;
                 batch_index < batch;
                 ++batch_index) {
                auto &scratch = scratches[batch_index];
                backend.normalize_rms(
                    scratch.hidden,
                    descriptor.input_norm,
                    1.0F,
                    1.0e-6F,
                    scratch.normalized);
                std::copy(
                    scratch.normalized.begin(),
                    scratch.normalized.end(),
                    batch_scratch.projection_5.begin() +
                        batch_index * config.hidden);
                positions[batch_index] = states[batch_index].position;
                full_states[batch_index] =
                    &states[batch_index].full_attention[layer];
                full_scratches[batch_index] =
                    &scratch.full_attention;
            }
            batch_scratch.projection_4.resize(batch * config.hidden);
            full_attention_forward_batch(
                model,
                layer,
                positions,
                batch_scratch.projection_5,
                full_states,
                full_scratches,
                batch_scratch.projection_4,
                batch_scratch,
                {},
                {},
                backend);
            batch_scratch.head_inputs.resize(batch * config.hidden);
            for (std::size_t batch_index = 0;
                 batch_index < batch;
                 ++batch_index) {
                auto &scratch = scratches[batch_index];
                const auto attention = std::span<const float>(
                    batch_scratch.projection_4)
                                           .subspan(
                                               batch_index * config.hidden,
                                               config.hidden);
                for (std::size_t index = 0;
                     index < scratch.hidden.size();
                     ++index) {
                    scratch.hidden[index] += attention[index];
                }
                backend.normalize_rms(
                    scratch.hidden,
                    descriptor.post_attention_norm,
                    1.0F,
                    1.0e-6F,
                    scratch.normalized);
                std::copy(
                    scratch.normalized.begin(),
                    scratch.normalized.end(),
                    batch_scratch.head_inputs.begin() +
                        batch_index * config.hidden);
            }
            batch_scratch.head_outputs.resize(batch * config.hidden);
            moe_forward_batch(
                model,
                layer,
                batch_scratch.head_inputs,
                batch_scratch.head_outputs,
                batch_scratch,
                backend);
            for (std::size_t batch_index = 0;
                 batch_index < batch;
                 ++batch_index) {
                auto &scratch = scratches[batch_index];
                for (std::size_t index = 0;
                     index < scratch.hidden.size();
                     ++index) {
                    scratch.hidden[index] +=
                        batch_scratch.head_outputs[
                            batch_index * config.hidden + index];
                }
            }
        } else {
            batch_scratch.head_inputs.resize(batch * config.hidden);
            for (std::size_t batch_index = 0;
                 batch_index < batch;
                 ++batch_index) {
                auto &scratch = scratches[batch_index];
                backend.normalize_rms(
                    scratch.hidden,
                    descriptor.input_norm,
                    1.0F,
                    1.0e-6F,
                    scratch.normalized);
                std::copy(
                    scratch.normalized.begin(),
                    scratch.normalized.end(),
                    batch_scratch.head_inputs.begin() +
                        batch_index * config.hidden);
                linear_states[batch_index] =
                    &states[batch_index].linear_attention[layer];
                linear_scratches[batch_index] =
                    &scratch.linear_attention;
            }
            batch_scratch.head_outputs.resize(batch * config.hidden);
            linear_attention_forward_batch(
                model,
                layer,
                batch_scratch.head_inputs,
                linear_states,
                linear_scratches,
                batch_scratch.head_outputs,
                batch_scratch,
                backend);
            for (std::size_t batch_index = 0;
                 batch_index < batch;
                 ++batch_index) {
                auto &scratch = scratches[batch_index];
                const auto attention = std::span<const float>(
                    batch_scratch.head_outputs)
                                           .subspan(
                                               batch_index * config.hidden,
                                               config.hidden);
                for (std::size_t index = 0;
                     index < scratch.hidden.size();
                     ++index) {
                    scratch.hidden[index] += attention[index];
                }
                backend.normalize_rms(
                    scratch.hidden,
                    descriptor.post_attention_norm,
                    1.0F,
                    1.0e-6F,
                    scratch.normalized);
                std::copy(
                    scratch.normalized.begin(),
                    scratch.normalized.end(),
                    batch_scratch.head_inputs.begin() +
                        batch_index * config.hidden);
            }
            batch_scratch.head_outputs.resize(batch * config.hidden);
            moe_forward_batch(
                model,
                layer,
                batch_scratch.head_inputs,
                batch_scratch.head_outputs,
                batch_scratch,
                backend);
            for (std::size_t batch_index = 0;
                 batch_index < batch;
                 ++batch_index) {
                auto &scratch = scratches[batch_index];
                for (std::size_t index = 0;
                     index < scratch.hidden.size();
                     ++index) {
                    scratch.hidden[index] +=
                        batch_scratch.head_outputs[
                            batch_index * config.hidden + index];
                }
            }
        }
    }

    batch_scratch.head_inputs.resize(batch * config.hidden);
    for (std::size_t batch_index = 0; batch_index < batch; ++batch_index) {
        finalize_hidden(model, scratches[batch_index], backend);
        ++states[batch_index].position;
        std::copy(
            scratches[batch_index].normalized.begin(),
            scratches[batch_index].normalized.end(),
            batch_scratch.head_inputs.begin() + batch_index * config.hidden);
    }

    for (std::uint32_t chunk = 0; chunk < 8; ++chunk) {
        const auto head = model.head_chunk(chunk);
        batch_scratch.head_outputs.resize(batch * head.rows);
        backend.head_matmul(
            head,
            batch_scratch.head_inputs,
            static_cast<std::uint32_t>(batch),
            batch_scratch.head_outputs);
        for (std::size_t batch_index = 0; batch_index < batch; ++batch_index) {
            const auto source = std::span<const float>(
                batch_scratch.head_outputs)
                                    .subspan(batch_index * head.rows, head.rows);
            auto destination = logits.subspan(
                batch_index * config.vocabulary +
                    static_cast<std::size_t>(chunk) * head.rows,
                head.rows);
            std::copy(source.begin(), source.end(), destination.begin());
        }
    }
}

void prefill(
    const MachModel &model,
    std::span<const std::uint32_t> tokens,
    DecoderState &state,
    std::span<float> logits,
    PrefillScratch &scratch,
    const Backend &backend) {
    const auto &config = model.config();
    if (tokens.empty() || logits.size() != config.vocabulary ||
        state.position >= config.context ||
        tokens.size() > config.context - state.position) {
        throw std::invalid_argument("prefill shape or context is invalid");
    }
    if (std::any_of(tokens.begin(), tokens.end(), [&](std::uint32_t token) {
            return token >= config.vocabulary;
        })) {
        throw std::invalid_argument("prefill token is out of range");
    }
    const auto count = tokens.size();
    const auto initial_position = state.position;
    scratch.hidden.resize(count * config.hidden);
    scratch.rope_cosine.resize(count * 32);
    scratch.rope_sine.resize(count * 32);
    const auto divisors = model.rope_theta_divisors();
    for (std::size_t token_index = 0; token_index < count; ++token_index) {
        initialize_hidden(model, tokens[token_index], scratch.token, backend);
        std::copy(
            scratch.token.hidden.begin(),
            scratch.token.hidden.end(),
            scratch.hidden.begin() + token_index * config.hidden);
        const auto position =
            initial_position + static_cast<std::uint32_t>(token_index);
        for (std::size_t index = 0; index < 32; ++index) {
            const float theta =
                static_cast<float>(position) / divisors[index];
            scratch.rope_cosine[token_index * 32 + index] = std::cos(theta);
            scratch.rope_sine[token_index * 32 + index] = std::sin(theta);
        }
    }

    std::vector<std::uint32_t> positions(count);
    std::vector<FullAttentionState *> full_states(count);
    std::vector<FullAttentionScratch *> full_scratches(count);
    std::vector<LinearAttentionState *> linear_states(count);
    std::vector<LinearAttentionScratch *> linear_scratches(count);
    for (std::size_t token_index = 0; token_index < count; ++token_index) {
        positions[token_index] =
            initial_position + static_cast<std::uint32_t>(token_index);
    }
    for (std::uint32_t layer = 0; layer < config.layers; ++layer) {
        const auto &descriptor = model.layer(layer);
        if (descriptor.full_attention) {
            scratch.batch.projection_5.resize(count * config.hidden);
            for (std::size_t token_index = 0;
                 token_index < count;
                 ++token_index) {
                std::copy_n(
                    scratch.hidden.begin() + token_index * config.hidden,
                    config.hidden,
                    scratch.token.hidden.begin());
                backend.normalize_rms(
                    scratch.token.hidden,
                    descriptor.input_norm,
                    1.0F,
                    1.0e-6F,
                    scratch.token.normalized);
                std::copy(
                    scratch.token.normalized.begin(),
                    scratch.token.normalized.end(),
                    scratch.batch.projection_5.begin() +
                        token_index * config.hidden);
                full_states[token_index] =
                    &state.full_attention[layer];
                full_scratches[token_index] =
                    &scratch.token.full_attention;
            }
            scratch.batch.projection_4.resize(count * config.hidden);
            full_attention_forward_batch(
                model,
                layer,
                positions,
                scratch.batch.projection_5,
                full_states,
                full_scratches,
                scratch.batch.projection_4,
                scratch.batch,
                scratch.rope_cosine,
                scratch.rope_sine,
                backend);
            scratch.batch.head_inputs.resize(count * config.hidden);
            for (std::size_t token_index = 0;
                 token_index < count;
                 ++token_index) {
                std::copy_n(
                    scratch.hidden.begin() + token_index * config.hidden,
                    config.hidden,
                    scratch.token.hidden.begin());
                const auto attention = std::span<const float>(
                    scratch.batch.projection_4)
                                           .subspan(
                                               token_index * config.hidden,
                                               config.hidden);
                for (std::size_t index = 0;
                     index < scratch.token.hidden.size();
                     ++index) {
                    scratch.token.hidden[index] += attention[index];
                }
                backend.normalize_rms(
                    scratch.token.hidden,
                    descriptor.post_attention_norm,
                    1.0F,
                    1.0e-6F,
                    scratch.token.normalized);
                std::copy(
                    scratch.token.normalized.begin(),
                    scratch.token.normalized.end(),
                    scratch.batch.head_inputs.begin() +
                        token_index * config.hidden);
                std::copy(
                    scratch.token.hidden.begin(),
                    scratch.token.hidden.end(),
                    scratch.hidden.begin() +
                        token_index * config.hidden);
            }
            scratch.batch.head_outputs.resize(count * config.hidden);
            moe_forward_batch(
                model,
                layer,
                scratch.batch.head_inputs,
                scratch.batch.head_outputs,
                scratch.batch,
                backend);
            for (std::size_t token_index = 0;
                 token_index < count;
                 ++token_index) {
                std::copy_n(
                    scratch.hidden.begin() + token_index * config.hidden,
                    config.hidden,
                    scratch.token.hidden.begin());
                for (std::size_t index = 0;
                     index < scratch.token.hidden.size();
                     ++index) {
                    scratch.token.hidden[index] +=
                        scratch.batch.head_outputs[
                            token_index * config.hidden + index];
                }
                std::copy(
                    scratch.token.hidden.begin(),
                    scratch.token.hidden.end(),
                    scratch.hidden.begin() +
                        token_index * config.hidden);
            }
        } else {
            scratch.batch.head_inputs.resize(count * config.hidden);
            for (std::size_t token_index = 0;
                 token_index < count;
                 ++token_index) {
                std::copy_n(
                    scratch.hidden.begin() + token_index * config.hidden,
                    config.hidden,
                    scratch.token.hidden.begin());
                backend.normalize_rms(
                    scratch.token.hidden,
                    descriptor.input_norm,
                    1.0F,
                    1.0e-6F,
                    scratch.token.normalized);
                std::copy(
                    scratch.token.normalized.begin(),
                    scratch.token.normalized.end(),
                    scratch.batch.head_inputs.begin() +
                        token_index * config.hidden);
                linear_states[token_index] =
                    &state.linear_attention[layer];
                linear_scratches[token_index] =
                    &scratch.token.linear_attention;
            }
            scratch.batch.head_outputs.resize(count * config.hidden);
            linear_attention_forward_batch(
                model,
                layer,
                scratch.batch.head_inputs,
                linear_states,
                linear_scratches,
                scratch.batch.head_outputs,
                scratch.batch,
                backend);
            for (std::size_t token_index = 0;
                 token_index < count;
                 ++token_index) {
                std::copy_n(
                    scratch.hidden.begin() + token_index * config.hidden,
                    config.hidden,
                    scratch.token.hidden.begin());
                const auto attention = std::span<const float>(
                    scratch.batch.head_outputs)
                                           .subspan(
                                               token_index * config.hidden,
                                               config.hidden);
                for (std::size_t index = 0;
                     index < scratch.token.hidden.size();
                     ++index) {
                    scratch.token.hidden[index] += attention[index];
                }
                backend.normalize_rms(
                    scratch.token.hidden,
                    descriptor.post_attention_norm,
                    1.0F,
                    1.0e-6F,
                    scratch.token.normalized);
                std::copy(
                    scratch.token.normalized.begin(),
                    scratch.token.normalized.end(),
                    scratch.batch.head_inputs.begin() +
                        token_index * config.hidden);
                std::copy(
                    scratch.token.hidden.begin(),
                    scratch.token.hidden.end(),
                    scratch.hidden.begin() +
                        token_index * config.hidden);
            }
            scratch.batch.head_outputs.resize(count * config.hidden);
            moe_forward_batch(
                model,
                layer,
                scratch.batch.head_inputs,
                scratch.batch.head_outputs,
                scratch.batch,
                backend);
            for (std::size_t token_index = 0;
                 token_index < count;
                 ++token_index) {
                std::copy_n(
                    scratch.hidden.begin() + token_index * config.hidden,
                    config.hidden,
                    scratch.token.hidden.begin());
                for (std::size_t index = 0;
                     index < scratch.token.hidden.size();
                     ++index) {
                    scratch.token.hidden[index] +=
                        scratch.batch.head_outputs[
                            token_index * config.hidden + index];
                }
                std::copy(
                    scratch.token.hidden.begin(),
                    scratch.token.hidden.end(),
                    scratch.hidden.begin() +
                        token_index * config.hidden);
            }
        }
    }

    std::copy_n(
        scratch.hidden.end() - config.hidden,
        config.hidden,
        scratch.token.hidden.begin());
    finalize_hidden(model, scratch.token, backend);
    for (std::uint32_t chunk = 0; chunk < 8; ++chunk) {
        const auto head = model.head_chunk(chunk);
        backend.head_matvec(
            head,
            scratch.token.normalized,
            logits.subspan(
                static_cast<std::size_t>(chunk) * head.rows,
                head.rows));
    }
    state.position += static_cast<std::uint32_t>(count);
}

} // namespace adi
