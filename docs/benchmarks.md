# CPU baseline

The first ADI baseline was measured on 2026-08-05 with a Release build on one
8-core AMD EPYC 9645 virtual machine (one thread per core). The model was the
7.07 GiB ADI GGUF produced from Mach-1-Additive-35B.

| Operation | Shape or scope | Time |
| --- | --- | ---: |
| Expert projection | 512 x 2048 | 8.56 ms |
| Non-expert projection | 512 x 2048 | 2.35 ms |
| Routed FFN | top-8 plus shared expert | 36.96 ms |
| Full attention | one cached token, one layer | 52.18 ms |
| Gated DeltaNet | one recurrent token, one layer | 65.84 ms |
| Output head | 31,040 x 2048 chunk | 18.76 ms |
| Complete decoder | 40 layers plus full head | 4.13 s/token |

The complete decoder reached 0.242 tokens/s, up from the initial scalar
0.0322 tokens/s baseline. These numbers are correctness-oriented starting
points, not tuned ceilings. The implementation uses simple row/tile
parallelism across eight cores; it does not yet use architecture-specific SIMD,
batched prompt evaluation, or NUMA tuning.

## Initial batch kernels

The first batching increment keeps single-vector kernels unchanged and decodes
packed weights once across several independent input vectors. On the same
machine, the per-vector costs were:

| Batch | Non-expert 512 x 2048 | Output head 31,040 x 2048 |
| ---: | ---: | ---: |
| 1 | 2.31 ms | 17.48 ms |
| 2 | 1.08 ms | 14.94 ms |
| 4 | 0.59 ms | 7.19 ms |
| 8 | 0.37 ms | 4.53 ms |

`decode_batch` currently uses the batched output head and preserves logits
bit-for-bit relative to independent `decode_token` calls. A layer-major
executor is still needed to route attention and shared-expert projections
through the non-expert batch kernel.

Reproduce the component suite:

```bash
tools/benchmark_cpu.sh models/Mach-1-Additive-35B.gguf
```

Add `--full` to include the slower full-token measurement.

## Runtime-dispatch and layer-major increment

Benchmark commands now use deterministic integer-hash inputs rather than
libm-generated vectors. They also print per-kernel call, elapsed-nanosecond,
and work-item counters. `tools/benchmark_cpu.sh` records the CPU, compiler,
`ADI_THREADS`, and `ADI_CPU_ISA` settings.

The following single-run Release measurements used GCC 14.2, eight workers,
and the same EPYC 9645 machine:

| Operation | Scalar | AVX-512 | Result |
| --- | ---: | ---: | --- |
| Int5 output-head chunk, 31,040 × 2,048 | 26.77 ms | 4.17 ms | 6.4× |
| Non-expert projection, 512 × 2,048 | 1.69 ms | 2.05 ms | no measured SIMD gain |

The AVX-512 and scalar output-head runs have the same deterministic checksum.
Random kernel tests bound BF16/int5 lane-reduction differences. Trellis
extraction, SIMD Hadamard stages, batch decoding, prompt logits, KV state, and
recurrent state remain bit-exact against their scalar or serial references.

A warmed complete token measured 3.64 seconds (0.274 token/s), versus the prior
recorded 4.13 seconds. The 11-token prompt plus one greedy output-token
checkpoint completes in roughly 12.5 seconds through layer-major prefill.
These are regression observations, not final throughput claims; repeat runs
and report variance before publishing tuned numbers.

## Prefill and batch-decode increment

Measured on 2026-08-07 with a Release build, GCC 14.2, eight EPYC 9645 cores
(one thread per core), AVX-512 selected, and the model warm in the page cache.
`adi bench-prefill MODEL TOKENS UBATCH [ITERATIONS]` reports these numbers with
a generated token sequence, a fresh `DecoderState` per measured iteration, and
FNV-1a checksums over the final logits and the complete decoder state.

### Headline

A 256-token prompt, each build at its own default microbatch, so this is what
a caller actually sees:

| | seconds | tokens/s | peak scratch |
| --- | ---: | ---: | ---: |
| Before, microbatch 64 | 171.57 | 1.49 | 15.4 MB |
| After, microbatch 16 | 50.58 | 5.06 | 8.0 MB |

3.39x the prompt throughput on 48% of the scratch, with logits checksum
`0x35a5f9bc4d1133d2` and state checksum `0xd3861b9b0d49fdd` on both.

Scratch is lower only because the default microbatch fell. Compared at the
same microbatch of 64 it rises from 15.4 MB to 31.8 MB, because the chunk-wide
linear-attention buffers and the persistent MoE stage buffers are new and both
scale with the microbatch. That is a real steady-state increase for anyone who
raises `--ubatch` back to 64 or beyond.

### What each change is worth

A 64-token prompt at microbatch 64, so every row is like-for-like. Kernel times
are the wall time attributed to that kernel across the whole prefill.

| Stage | tokens/s | non-expert batch | linear attention | MoE |
| --- | ---: | ---: | ---: | ---: |
| Configurable microbatch only | 1.515 | 22.21 s | 16.42 s | 21.42 s |
| Non-expert batch kernel | 3.126 | 0.57 s | 0.81 s | 19.49 s |
| Linear-attention prefill chunk | 3.157 | 0.65 s | 0.67 s | 19.42 s |
| Batched MoE working set | 3.176 | 0.65 s | 0.66 s | 19.31 s |

The non-expert batch kernel is nearly the whole prefill gain. It also accounts
for most of the linear-attention drop, because that layer's cost was its
non-expert projections rather than the recurrence or the per-token worker
dispatch. Restructuring linear attention into one dispatch per chunk is worth
18% of that kernel and about 1% end to end.

Every row above produces logits checksum `0x8c24a3f4644e03d3` and state
checksum `0xc75f92166aad6875`.

### Microbatch sweep

Superseded by the sweep under "Expert dispatch scheduling" below: splitting
expert work into twelve-row chunks inverted this curve. The table records what
this increment measured, not the current default.

A 256-token prompt. Peak scratch is the summed capacity of every prefill
scratch buffer after the run.

| Microbatch | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tokens/s | 1.99 | 3.35 | 4.40 | 5.06 | 4.25 | 3.05 | 2.67 | 2.45 |
| peak scratch (MB) | 1.0 | 2.0 | 4.1 | 8.0 | 16.0 | 31.8 | 63.5 | 126.9 |

Sixteen is fastest and cheapest, so it is the default. A 1024-token prompt
keeps the ordering: 4.89 tokens/s at sixteen against 3.00 at sixty-four.

Large microbatches were expected to win, since decoding a packed weight once
across many vectors is what made the non-expert kernel fast. They lose because
of the other codec: once several tokens route to the same expert,
`mach_expert_matmul` leaves the single-vector path for a scalar batch loop
whose row accumulators and strided inputs no longer fit L1. Giving that kernel
the same batch-lane treatment is the next piece of work, and this default
should be re-measured after it.

All eight microbatches produce the same checksums at a given prompt length.

### Decode

`adi decode-batch`, sequences/second. Batches 4, 8, and 16 are medians of seven
runs; batches 1 and 32 are medians of three. Run-to-run spread is roughly
±10%, so single runs do not separate these.

| Batch | Before | After |
| ---: | ---: | ---: |
| 1 | 1.29 | 1.97 |
| 4 | 2.67 | 4.25 |
| 8 | 3.04 | 5.89 |
| 16 | 3.22 | 6.69 |
| 32 | 2.44 | 4.60 |

Aggregate throughput used to flatten between batch 4 and batch 8. The
non-expert batch kernel fixed that on its own, taking the pair from 2.67/3.04
to 4.37/5.69; the batched MoE working set then added about 7% at batches 4
through 16. Throughput still falls from batch 16 to batch 32, which is the same
expert-codec effect the microbatch sweep shows.

Single-sequence decode improves from 1.29 to 1.97 tokens/s, well clear of the
2% no-regression requirement. It gains from the specialized trellis walk alone,
since that path uses no batch kernel.

### Exactness

Bit-exact, not tolerance-bounded:

- prompt lengths {1, 63, 64, 65, 257} against microbatches {1, 7, 64, 256}
  agree on final logits, position, and all KV, convolution, and recurrent
  state, and agree with token-at-a-time `decode_token`;
- the specialized non-expert trellis walk reproduces the generic bit-window
  extractor on 4,000 randomized tiles plus every single-bit edge case,
  including the transition that wraps the stream;
- the SIMD batch kernel equals the scalar kernel at batches {1, 2, 3, 4, 5, 7,
  8, 15, 16, 17, 33, 64, 127} on three matrix shapes under both AVX2 and
  AVX-512, and each batch row equals the single-vector kernel;
- the linear-attention prefill chunk equals repeated `linear_attention_forward`
  at {1, 2, 7, 16, 64, 65} tokens from a nonzero state;
- batched MoE equals per-token `moe_forward` on routes, route weights, and
  outputs at batches {1, 4, 8, 64, 256} for identical, varied, and duplicated
  routing.

The batch kernels compile with FP contraction disabled so their separate
multiply and add are never fused; the objects contain no FMA instruction. The
scalar path is retained and still runs below batch 4 and on ISAs without a
batch kernel. The full suite passes under `ADI_CPU_ISA` of scalar, avx2, and
avx512.

## Expert dispatch scheduling

Measured on 2026-08-07, same machine and build as the section above.

The batched MoE was 96% of prefill time, and only 47% of the worker time
inside its expert dispatch was spent computing. Routing is skewed: a layer has
about 85 active experts holding 512 rows between them, roughly a third holding
one row while the largest holds around 56. Contiguous task blocks handed that
out by position rather than by cost.

`mach_expert_matmul` also reads its input column-major over a row-major array,
so its cost per row depends on how many rows it is given:

| rows | 2 | 4 | 8 | 12 | 16 | 24 | 32 | 48 | 64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ms/row | 1.51 | 0.97 | 0.66 | 0.55 | 0.72 | 0.91 | 1.15 | 1.52 | 1.88 |

Splitting a large expert into evenly sized chunks of at most twelve rows is
therefore faster in absolute terms as well as better balanced, and
`parallel_dynamic` hands the chunks out on demand so imbalance is bounded by
one chunk.

| | before | after |
| --- | ---: | ---: |
| prefill, 64 tokens at microbatch 64 | 20.11 s | 7.09 s |
| tokens/s | 3.18 | 9.03 |
| expert-kernel CPU | 71.7 s | 46.6 s |
| dispatch parallel efficiency | 0.47 | 0.933 |

Decode, medians of seven runs, sequences/second:

| batch | 1 | 4 | 8 | 16 | 32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| before | 1.97 | 4.25 | 5.89 | 6.69 | 4.60 |
| after | 1.97 | 4.83 | 6.16 | 6.29 | 7.45 |

Throughput now climbs from batch 4 through 32 rather than peaking at 16. Batch
1 is unchanged because single-sequence decode uses `moe_forward`, not the
batched path. Batch 16 is nominally lower but the two ranges overlap.

Checksums are unchanged and the wide microbatch matrix is exact in all twenty
cases. The twelve-row cap exists only because of the expert kernel's current
batch behaviour; re-measure it if that kernel gains a batch-lane path.

### Microbatch after expert-dispatch splitting

Chunking expert work inverted the microbatch curve. The same 256-token prompt,
before and after that change:

| Microbatch | 16 | 32 | 64 |
| --- | ---: | ---: | ---: |
| tokens/s, one matmul per expert | 5.06 | 4.25 | 3.05 |
| tokens/s, twelve-row chunks | 6.31 | 7.70 | 8.40 |
| peak scratch (MB) | 8.0 | 16.0 | 31.8 |

`ExecutionOptions::prefill_ubatch` moves from 16 to 64, which is 33% more
prompt throughput for 31.8 MB of prefill scratch against 8.0 MB. Callers who
want the smaller footprint set `--ubatch 16`; the choice never changes what a
request returns.

This is interim. `mach_expert_matmul` is still a scalar batch loop, so the
sweep above stops at 64 and the full range including 128 and beyond is measured
after that kernel gains a batch-lane path.
