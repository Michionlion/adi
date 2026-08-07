#pragma once

#include "adi/backend.hpp"
#include "adi/model.hpp"

#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace adi {

struct ExpertRoute {
    std::uint32_t expert;
    float weight;
};

struct MoeExpertScratch {
    ExpertScratch codec;
    std::vector<float> gate;
    std::vector<float> up;
    std::vector<float> activated;
    std::vector<float> projected;
};

struct MoeScratch {
    std::array<MoeExpertScratch, 8> experts;
    ExpertScratch shared_codec;
    std::vector<float> router_logits;
    std::vector<float> shared_gate;
    std::vector<float> shared_up;
    std::vector<float> shared_down;
};

struct FullAttentionState {
    std::vector<float> keys;
    std::vector<float> values;
};

struct FullAttentionScratch {
    ExpertScratch codec;
    std::vector<float> query_gate;
    std::vector<float> contiguous_queries;
    std::vector<float> key;
    std::vector<float> value;
    std::vector<float> attended;
    std::vector<float> scores;
    std::uint32_t rope_position = std::numeric_limits<std::uint32_t>::max();
    std::array<float, 32> rope_cosine;
    std::array<float, 32> rope_sine;
};

struct LinearAttentionState {
    std::vector<float> convolution;
    std::vector<float> recurrent;
};

struct LinearAttentionScratch {
    ExpertScratch codec;
    std::vector<float> qkv;
    std::vector<float> gate;
    std::vector<float> convolved;
    std::vector<float> alpha;
    std::vector<float> beta;
    std::vector<float> recurrent_output;
    std::vector<float> normalized;
};

// Chunk-wide buffers for the prefill linear-attention kernel. Each vector
// holds one phase's values for every token in the chunk, laid out
// token-major: [token, channels] for qkv and convolved, [token, value_size]
// for gate, recurrent_output, and normalized, [token, value_heads] for the
// decay inputs.
struct LinearPrefillScratch {
    ExpertScratch codec;
    std::vector<float> qkv;
    std::vector<float> gate;
    std::vector<float> alpha;
    std::vector<float> beta;
    std::vector<float> convolved;
    std::vector<float> recurrent_output;
    std::vector<float> normalized;
};

struct DecoderState {
    std::uint32_t position = 0;
    std::array<FullAttentionState, 40> full_attention;
    std::array<LinearAttentionState, 40> linear_attention;
};

struct DecoderScratch {
    std::vector<float> hidden;
    std::vector<float> normalized;
    std::vector<float> attention;
    std::vector<float> feed_forward;
    FullAttentionScratch full_attention;
    LinearAttentionScratch linear_attention;
    MoeScratch moe;
};

// Persistent working set for the batched MoE. Every buffer is resized and
// reused, never reconstructed, so a steady-size workload stops allocating
// after the first call.
//
// Routes are grouped by a counting sort: each active expert owns one
// contiguous range [offsets[expert], offsets[expert] + counts[expert]) of the
// stage buffers. route_to_grouped maps a (batch, route) pair back to its
// position in that grouping, which is what lets the output reduction run in
// route order rather than grouped order.
struct MoeBatchScratch {
    std::vector<float> router_logits;
    std::vector<std::uint32_t> route_order;
    std::vector<std::array<ExpertRoute, 8>> routes;
    std::vector<std::uint32_t> counts;
    std::vector<std::uint32_t> offsets;
    std::vector<std::uint32_t> cursors;
    std::vector<std::uint32_t> active;
    std::vector<std::uint32_t> grouped_batch;
    std::vector<std::uint32_t> route_to_grouped;
    std::vector<float> gathered;
    std::vector<float> gate;
    std::vector<float> up;
    std::vector<float> activated;
    std::vector<float> projected;
};

struct DecoderBatchScratch {
    ExpertScratch codec;
    MoeBatchScratch moe;
    std::vector<float> head_inputs;
    std::vector<float> head_outputs;
    std::vector<float> projection_0;
    std::vector<float> projection_1;
    std::vector<float> projection_2;
    std::vector<float> projection_3;
    std::vector<float> projection_4;
    std::vector<float> projection_5;
};

struct PrefillScratch {
    std::vector<float> hidden;
    std::vector<float> rope_cosine;
    std::vector<float> rope_sine;
    DecoderScratch token;
    DecoderBatchScratch batch;
    LinearPrefillScratch linear;
};

[[nodiscard]] std::array<ExpertRoute, 8> top_experts(
    std::span<const float> logits);

// Same routing, with the index scratch supplied by the caller so a hot loop
// does not allocate one per token.
[[nodiscard]] std::array<ExpertRoute, 8> top_experts(
    std::span<const float> logits,
    std::vector<std::uint32_t> &order);

[[nodiscard]] std::uint32_t gated_delta_key_head(
    std::uint32_t value_head,
    std::uint32_t value_heads,
    std::uint32_t key_heads);

[[nodiscard]] std::array<ExpertRoute, 8> moe_forward(
    const MachModel &model,
    std::uint32_t layer,
    std::span<const float> input,
    std::span<float> output,
    MoeScratch &scratch,
    const Backend &backend = cpu_backend());

// Evaluates the MoE block for a batch of independent vectors. Inputs and
// outputs are [batch, hidden]. Each row's routes, route weights, and output
// match moe_forward exactly. Routes are left in batch_scratch.moe.routes.
void moe_forward_batch(
    const MachModel &model,
    std::uint32_t layer,
    std::span<const float> inputs,
    std::span<float> outputs,
    DecoderBatchScratch &batch_scratch,
    const Backend &backend = cpu_backend());

void full_attention_forward(
    const MachModel &model,
    std::uint32_t layer,
    std::uint32_t position,
    std::span<const float> input,
    std::span<float> output,
    FullAttentionState &state,
    FullAttentionScratch &scratch,
    const Backend &backend = cpu_backend());

void linear_attention_forward(
    const MachModel &model,
    std::uint32_t layer,
    std::span<const float> input,
    std::span<float> output,
    LinearAttentionState &state,
    LinearAttentionScratch &scratch,
    const Backend &backend = cpu_backend());

// Evaluates one chunk of a single sequence. Inputs and outputs are
// [tokens, hidden]. The convolution and recurrent state advance exactly as
// the same number of linear_attention_forward calls would, but the worker
// pool is dispatched once for the chunk rather than once per token.
void linear_attention_prefill_chunk(
    const MachModel &model,
    std::uint32_t layer,
    std::span<const float> inputs,
    std::uint32_t tokens,
    LinearAttentionState &state,
    LinearPrefillScratch &scratch,
    std::span<float> outputs,
    const Backend &backend = cpu_backend());

void decode_token(
    const MachModel &model,
    std::uint32_t token,
    DecoderState &state,
    // An empty span advances the model state without computing unused logits.
    std::span<float> logits,
    DecoderScratch &scratch,
    const Backend &backend = cpu_backend());

// Decodes one token for each independent sequence. Logits are batch-major
// [tokens.size(), model.config().vocabulary].
void decode_batch(
    const MachModel &model,
    std::span<const std::uint32_t> tokens,
    std::span<DecoderState> states,
    std::span<float> logits,
    std::span<DecoderScratch> scratches,
    DecoderBatchScratch &batch_scratch,
    const Backend &backend = cpu_backend());

// Evaluates one sequence in layer-major order. Only the final prompt token
// produces logits; all recurrent and KV state is retained for decode.
void prefill(
    const MachModel &model,
    std::span<const std::uint32_t> tokens,
    DecoderState &state,
    // An empty span advances the model state without computing unused logits.
    std::span<float> logits,
    PrefillScratch &scratch,
    const Backend &backend = cpu_backend());

} // namespace adi
