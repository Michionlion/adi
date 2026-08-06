# ADI CPU optimization plan

## Current measured baseline

ADI's repository records the following Release measurements on eight AMD EPYC 9645 cores:

| Operation | Time |
| --- | ---: |
| Expert projection, 512 × 2048 | 8.56 ms |
| Non-expert projection, 512 × 2048 | 2.35 ms |
| Routed FFN, top-8 plus shared expert | 36.96 ms |
| Full-attention layer, one cached token | 52.18 ms |
| Gated DeltaNet layer, one recurrent token | 65.84 ms |
| Output-head chunk, 31,040 × 2048 | 18.76 ms |
| Complete 40-layer decoder plus head | 4.13 s/token, 0.242 token/s |

The repository also measures non-expert projection cost per vector falling from 2.31 ms at batch 1 to 0.37 ms at batch 8, and output-head cost per vector falling from 17.48 ms to 4.53 ms. These batch results are direct evidence that weight reuse is valuable. Estimated gains below are directional and are not additive.

## P0: highest-impact work

### 1. SIMD packed expert and non-expert kernels

The dominant codec loops are scalar and repeatedly perform bit extraction, lookup/sign reconstruction, FP16 round trips, and scalar Hadamard transforms.

Implement:

- direct fixed-width extraction of trellis fresh bits rather than one-bit loops;
- predecoded state/component metadata per tile format;
- exact FP16-rounding lookup caches or vectorized F16C conversions, without constructing a dense weight shadow;
- AVX2 and AVX-512 row-lane kernels on x86;
- NEON/SVE kernels on Arm;
- vectorized Hadamard stages;
- runtime CPU-feature dispatch, with the scalar path retained as the reference implementation.

Correctness constraint: preserve the released Python decoder's rounding and accumulation contract. Compare every optimized kernel against codec goldens and randomized scalar-reference vectors.

Likely payoff: roughly 3–8× for the packed kernels and 2–4× end-to-end, subject to profiling.

### 2. Layer-major batched execution and real prompt prefill

`decode_batch` currently advances each sequence through all 40 layers independently and batches primarily the output head. Reorganize execution by layer so packed weights are decoded once across multiple sequence vectors.

Implement:

- batch non-expert projections throughout attention and the shared expert;
- group routed tokens by expert and batch gate/up/down expert projections;
- continuous request batching in the server;
- block prompt prefill rather than token-at-a-time decode;
- chunked Gated DeltaNet prefill matching the Qwen reference algorithm;
- block/full-attention prefill using matrix-matrix operations and online softmax.

The measured batch-8 non-expert result is a 6.2× per-vector improvement over batch 1. A full layer-major implementation should materially improve prefill and multi-request throughput; 3–6× is a reasonable engineering target to test, not a guaranteed result.

## P1: substantial, lower-risk wins

### 3. Persistent worker pool

The decoder repeatedly creates and joins native threads in routed MoE and row-parallel kernels. Replace per-operation thread creation with one fixed, configurable worker pool and request-local task groups.

Expected benefit: lower token latency and less scheduler noise, especially on 8–32 core systems. Benchmark target: 5–15% end-to-end.

### 4. Cache validated model descriptors

Runtime accessors repeatedly build tensor-name strings, linearly search GGUF descriptors, and reconstruct validated spans. Build immutable per-layer descriptors once during model construction containing every matrix/vector view and geometry.

Expected benefit: approximately 5–20%, depending on the current lookup share, plus simpler hot-path code.

### 5. Precompute RoPE frequencies and per-position sin/cos

Full attention currently repeats `pow`, `sin`, and `cos` for every query/key head and every full-attention layer. Store inverse frequencies once and compute one 32-pair sin/cos table per position, then reuse it across all heads and layers.

This is an unusually straightforward latency win and should precede more invasive attention work.

### 6. Vectorize the int5 output head

The output head unpacks int5 codes and converts group scales in scalar inner loops. Implement ISA-specific unpack/dot kernels, hoist one scale conversion per 64-element group, and process multiple rows together.

Likely payoff: 5–10× for the head kernel. At the recorded baseline, even eliminating most head time saves only part of total token latency, so this should accompany codec work rather than replace it.

### 7. Grouped-query online-softmax attention decode

Eight query heads share each KV head, but the current implementation walks the same key/value history separately. Process the query group together with one KV read stream and an online-softmax accumulator.

At long context this should reduce memory traffic substantially; a 4–8× reduction in attention-state work is a useful target to validate.

### 8. Vectorize small BF16 dense projections and simplify routing

Add AVX-512 BF16 or vectorized BF16-to-FP32 kernels for routers and Gated DeltaNet `a`/`b` projections. For MoE routing, select the top eight logits first and normalize only those eight: the global softmax denominator cancels when selected probabilities are renormalized.

This is exact and removes unnecessary exponentials and allocation from every layer.

### 9. Reduce sampling overhead

Non-greedy sampling sorts all 248,320 vocabulary indexes every token. Profile this separately from the output head. Exact alternatives include float-key radix sorting or a selection strategy that avoids comparison-sorting the full vocabulary while still honoring top-p semantics.

Do not introduce an arbitrary top-k cap unless it is an explicit user-visible sampling option, because that changes results.

## P2: platform and memory work

### 10. KV-cache allocation and long-context memory layout

Use reserved or paged KV storage to avoid vector growth/copying. Lay out grouped KV heads for sequential decode access and consider cache-line alignment and software prefetching.

### 11. NUMA and mmap policy

On multi-socket EPYC systems:

- pin workers and allocate scratch/state near their workers;
- choose a deliberate model-page placement policy;
- test `madvise` strategies and transparent huge pages;
- measure page faults and remote-memory traffic rather than assuming a policy helps.

### 12. LTO, PGO, and reproducible ISA builds

Add optional LTO and profile-guided builds after stable microbenchmarks exist. Prefer runtime-dispatched object files over a single `-march=native` binary so release artifacts remain portable.

### 13. Speculative/MTP decoding only after the core path is fast

MTP or speculative decoding can improve accepted tokens per model evaluation, but a full MoE draft head can itself be expensive. Treat it as a later system-level optimization after packed kernels, batching, and memory scheduling are efficient.

## Suggested implementation order

1. Add per-kernel profiling counters and stable benchmark inputs.
2. Cache all model descriptors and precompute RoPE data.
3. Introduce the persistent worker pool.
4. Implement scalar-structured codec refactors that preserve exact output.
5. Add AVX2/AVX-512 and NEON/SVE codec dispatch.
6. Vectorize the output head and BF16 projections.
7. Build layer-major prefill and continuous batching.
8. Add grouped-query online-softmax attention.
9. Tune NUMA/mmap behavior and then evaluate LTO/PGO.
10. Consider MTP/speculative decoding.

## Validation requirements

For each optimization:

- compare against the scalar implementation on deterministic and randomized vectors;
- retain released codec golden tests;
- require batch-versus-serial state and logit equivalence where the operation order is intended to remain identical;
- for deliberate floating-point reorderings, report maximum/mean error, relative L2, top-token agreement, and end-to-end perplexity/KL measurements;
- record hardware, compiler, flags, thread affinity, warmup, and run variance with every performance claim.
