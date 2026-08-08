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

Superseded twice, by "Expert dispatch scheduling" and then by "Expert codec
batch lanes" below. Both of the reasons this curve peaked at sixteen have since
been removed, so the table records what this increment measured, not the
current default.

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

`ExecutionOptions::prefill_ubatch` moved from 16 to 64 on this measurement,
which was 33% more prompt throughput for 31.8 MB of prefill scratch against
8.0 MB. It moves again in the next section, once the expert codec stops
penalizing a wide microbatch at all.

## Expert codec batch lanes

Measured on 2026-08-07, same machine and build as the sections above: a
Release build, GCC 14.2, eight EPYC 9645 cores one thread per core, AVX-512
selected, and the model warm in the page cache. Every figure is the median of
seven runs, with the spread given where it decides anything.

`mach_expert_matmul` was the last scalar batch loop in the prefill. It decoded
a packed weight and then walked the batch dimension one multiply-add at a time,
reading `scratch.input[batch_index * columns + column]` column-major over a
row-major array, so its cost per row was U-shaped: 1.51, 0.66, 0.55, 0.72,
1.15, and 1.88 ms per row at 2, 8, 12, 16, 32, and 64 rows. The runtime worked
around that by capping expert tasks at twelve rows.

It now gets the treatment `ne_batch` already gave the K=4/V=2 non-expert
codec. Inputs pack to `[block][column][lane]`, each decoded weight broadcasts
across 8 (AVX2) or 16 (AVX-512) independent batch items, and only the batch
dimension is vectorized, never the reduction.

### What each change is worth

A 64-token prompt at microbatch 64, so every row is like-for-like. Kernel
columns are the wall time attributed to that kernel across the whole prefill.

| Stage | tokens/s | expert matvec | expert batch |
| --- | ---: | ---: | ---: |
| Before | 8.896 | 7.00 s | 39.34 s |
| Shared trellis walk | 9.653 | 2.84 s | 39.3 s |
| Expert batch lanes | 27.552 | 2.84 s | 7.06 s |
| Task cap at 32 rows | 29.843 | | |

Run-to-run spread was within ±1.6% on every row, so each step is far outside
it.

This is two separable changes and they are credited separately. Sharing one
trellis walk between `mach_expert_matvec` and `mach_expert_matmul` inlines the
twelve-bit window extraction over compile-time constants and replaces a
per-element division and modulo with a shift. That is worth 8.5% by itself,
and it lands on the single-vector kernel, which has no SIMD path and still
runs 4002 of the calls in a prefill. The batch lanes are worth 2.85x on top.

The expert kernel gains less than the non-expert kernel's 39x because this
codec has more per-state work to amortize: eight values per state against two,
wave indexes, a per-tile gamma, and a 2 MB state-value table rather than
512 KB.

### Expert kernel cost per row

Serial, one thread, the 512x2048 gate projection, which is the cost a single
`parallel_dynamic` task sees:

| rows | 1 | 2 | 4 | 8 | 12 | 16 | 32 | 64 | 128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ms/call | 0.75 | 0.70 | 0.68 | 0.75 | 0.75 | 0.80 | 1.27 | 2.76 | 5.18 |
| ms/row | 0.753 | 0.351 | 0.171 | 0.094 | 0.063 | 0.050 | 0.040 | 0.043 | 0.041 |

The U-shape is gone. A call costs about the same for any batch that fits one
SIMD block, so cost per row falls monotonically and splitting an expert is now
pure overhead beyond what load balance needs. `expert_rows_per_task` moves
from 12 to 32, two full blocks: 28.0 tokens/second at twelve, 29.2 at sixteen,
29.8 at thirty-two, 27.8 at sixty-four, and 28.3 with no cap at all. Past two
blocks the dispatch runs out of tasks to balance across eight workers.

Row 1 is `mach_expert_matvec`, which single-row experts still use and which
accounts for 4002 of the calls in a prefill. Routing them through the lane
kernel instead measured 29.783 against 29.388 tokens/second, inside the
run-to-run spread, so the simpler path stays. A single row has nothing to
vectorize across; the lane kernel would fill one lane of sixteen.

### Microbatch sweep, re-measured

The expert codec was the reason a wide microbatch lost. With that gone,
throughput is monotonic in the microbatch at both prompt lengths.

A 256-token prompt:

| Microbatch | 16 | 32 | 64 | 128 | 256 |
| --- | ---: | ---: | ---: | ---: | ---: |
| tokens/s | 18.16 | 23.85 | 29.05 | 33.28 | 37.14 |
| peak scratch (MB) | 8.0 | 15.9 | 31.8 | 63.5 | 126.9 |

A 1024-token prompt, which is what decides the default now that there is no
falloff to find:

| Microbatch | 64 | 128 | 256 | 512 | 1024 |
| --- | ---: | ---: | ---: | ---: | ---: |
| tokens/s | 25.57 | 29.33 | 32.43 | 34.67 | 36.06 |
| peak scratch (MB) | 31.8 | 63.5 | 126.9 | 253.6 | 507.2 |
| gain over previous | | +14.7% | +10.5% | +6.9% | +4.0% |

Every point is on the Pareto front, so the default is an exchange rate rather
than an optimum. Scratch doubles at every step while the gain shrinks, and 128
is the last step that buys more than 10%. `ExecutionOptions::prefill_ubatch`
moves from 64 to 128: 14.7% more prompt throughput for 63.5 MB of prefill
scratch against 31.8 MB.

Scratch is allocated for the microbatch actually used, which is the smaller of
the setting and the prompt length. A 64-token prompt costs 31.8 MB at
`--ubatch 512` exactly as it does at `--ubatch 64`, so only a caller sending
long prompts pays for a large value. On a machine with memory to spare and
long prompts to run, 256 or 512 is a reasonable `--ubatch`.

### Decode

`adi decode-batch`, sequences/second, medians of seven runs. Spread here is
much wider than in prefill, roughly ±10% and up to ±16% at batch 1, so treat
the smaller differences as ties.

| Batch | 1 | 4 | 8 | 16 | 32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Before | 1.97 | 4.83 | 6.16 | 6.29 | 7.45 |
| Expert lanes | 2.96 | 8.82 | 12.83 | 17.13 | 18.82 |
| Linear-attention dispatch | 2.94 | 10.29 | 17.03 | 27.20 | 31.69 |

Batch 1 gains 50% from the expert work without touching a batch kernel at all:
single-sequence decode routes through `moe_forward` and `mach_expert_matvec`,
so what it picks up is the shared trellis walk. Everything above batch 1 gains
from the lanes as well, and the curve no longer flattens after batch 16.

The second row is `linear_attention_forward_batch`, which batched only its
projections and ran the convolution, the normalizations, the recurrence and
the gating in a serial loop over sequences, dispatching the worker pool once
per sequence per layer. Dispatching once over sequences instead takes linear
attention from 1261 ms to 488 ms of the batch-32 decode step. Batch 1 is
unchanged because a single task runs in place on the calling thread, which is
not a worker, so the recurrence still spreads across its 32 value heads.

### Decode dispatch and thresholds

Three further decode changes, medians of seven runs, sequences/second:

| Batch | 1 | 2 | 3 | 4 | 8 | 16 | 32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| One dispatch per sequence | 2.94 | 4.01 | 5.24 | 10.29 | 17.03 | 27.20 | 31.69 |
| Sequences times heads | | | | 10.81 | 18.09 | 27.19 | 33.20 |
| Non-expert threshold of two | 2.91 | 5.42 | 7.87 | 10.32 | 16.72 | | |

Dispatching the recurrence over sequences and value heads together, rather
than one task per sequence, is worth a consistent 5% at batches 4, 8 and 32.
The individual ranges overlap, so the number to trust is the direction and
the structure it removes: one task per sequence used `min(threads, batch)`
cores, so batch three ran on three of eight.

`ne_batch_minimum` mattered much more. It was four, so batches two and three
fell off the non-expert SIMD kernel onto the scalar loop, and a decode step
cost more at batch two (0.499 s) than at batch four (0.388 s). A batch of two
occupies one SIMD block exactly as sixteen does, so the kernel costs one pass
over the tiles either way; the scalar loop costs a pass per batch item. On
the 1408x2048 shared-expert gate, serially, one pass is 0.93 ms against 1.60
at batch two and 1.77 at batch three. Lowering the threshold to two makes
step time monotonic in the batch again: 0.344, 0.369, 0.381, 0.388, 0.479
seconds at batches 1, 2, 3, 4 and 8.

### MoE dispatch under realistic routing

`adi decode-batch` gives every sequence the same token, so every sequence
routes to the same eight experts. That is not what serving looks like, and it
is not what the dispatch was tuned for. `adi bench-moe-batch ... varied`
routes them independently:

| | active experts | rows per expert | parallel efficiency |
| --- | ---: | ---: | ---: |
| identical, batch 8 | 8 | 8 | 0.61 |
| identical, batch 32 | 8 | 32 | 0.57 |
| varied, batch 8 | 50 | 1..3 | 0.88 |
| varied, batch 32 | 127 | 1..13 | 0.92 |

The dispatch is already well balanced where it counts. The low efficiency on
the identical pattern is eight equal tasks on eight workers with the router,
the counting sort, the gather, the serial route reduction and the shared
expert all outside the expert dispatch; it is not imbalance.

What realistic routing does show is that half the expert kernel time goes to
experts holding a single row, because 127 of 256 experts activate and most
hold one or two. Routing those through the batch-lane kernel rather than
`mach_expert_matvec` measured 0.01602 against 0.01549 seconds per forward at
batch 8 and 0.03859 against 0.04052 at batch 32 -- worse at one, better at
the other, both inside the spread -- so `expert_batch_minimum` stays at two.
Decode MoE is bound by streaming each active expert's packed weights once,
which no amount of row scheduling changes; only more tokens sharing an expert
does.

### Where decode time goes now

Wall time per decode step, from the stage timers:

| | batch 1 | batch 8 | batch 32 |
| --- | ---: | ---: | ---: |
| linear attention | 170 ms | 176 ms | 488 ms |
| MoE | 143 ms | 159 ms | 317 ms |
| full attention | 42 ms | 44 ms | 73 ms |
| step total | 367 ms | 441 ms | 1044 ms |

Two things follow. Decode is latency-bound rather than throughput-bound below
batch 8: a step costs 367 ms for one sequence and 441 ms for eight, so the
batch is nearly free. And linear attention is still the largest single stage
at every batch, so it is where the next decode work belongs.

### The single-vector non-expert kernel

Linear attention being the largest decode stage does not mean the recurrence
is what costs. `adi bench-linear MODEL 0 1`, one layer, one token:

| | ms | share |
| --- | ---: | ---: |
| three non-expert projections | 5.228 | 95.1% |
| convolution, recurrence, normalizations, gating | 0.255 | 4.6% |

The gated-delta recurrence is not the problem. `mach_ne_matvec` is. Its
`KernelTimer` runs on the calling thread and wraps its own `parallel_ranges`,
so its total across 250 calls is directly comparable to the step: 188 ms of a
332 ms token, or **57% of a batch-1 decode step**. It moves 6.4 million weight
elements per millisecond because its accumulation is scalar per element.

It has no SIMD path because the obvious axis is illegal: the eight states of
a tile row accumulate into one scalar, and reassociating that reduction would
change the result. There is a legal axis. State i of a non-expert tile lands
in local row `i >> 3` at local columns `(i & 7) * 2` and `+ 1`, so for a
fixed step every one of the 16 rows in a tile reads the *same* pair of input
columns while drawing its own pair of weights from its own state. Sixteen
rows are sixteen independent accumulators, which is exactly one AVX-512
register: broadcast the two input scalars, gather the sixteen weight pairs,
and each lane performs its row's adds in the scalar kernel's order.

That kernel was built and measured. It is bit-exact -- the whole suite passes
under it on both AVX2 and AVX-512, including the existing check that every
batch row of the SIMD matmul equals the single-vector kernel -- and it is not
faster. One linear-attention layer went from 5.228 to 5.195 ms, and the
non-expert total over a decode step from 188.3 to 184.7 ms. It is reverted.

The reason is the gather. Sixteen rows need sixteen different weight pairs
drawn from sixteen different states, so each step needs
`vgatherdps`, and a tile gathers 256 elements -- exactly the 256 values the
scalar path loads. Gather throughput on this core is about one element per
cycle, against two scalar loads per cycle, so vectorizing the arithmetic
removes work that was never the constraint. Shrinking the state table would
not change it either: the table is a single shared 512 KB array, not one per
matrix, and the cost is the instruction rather than the miss.

Hoisting the trellis walk out of the arithmetic was also tried, on the theory
that the walk's serial chain was blocking the state-table loads. It is worse:
2.687 against 3.003 sequences/second, and the non-expert total rises from 188
to 226 ms. Staging 128 states through a 256-byte array costs more than the
dependency it breaks.

Both results are worth knowing before anyone tries again. What would change
the picture is a lookup that is not a gather -- fewer, wider loads, or a
representation whose weights are computed rather than looked up.

That guess was wrong in an instructive way. Both of those were measured and
neither pays: see "The accumulator, not the lookup" below. The lookup was not
what the kernel was spending its time on, so making it cheaper could not help.

### Measuring batch-1 decode

Batch-1 decode is far noisier than prefill and noisier than the rest of
decode. Twenty-one runs of the same binary give a median of 2.770
sequences/second with a spread of -14.5% to +7.3%, and separate sessions of
*identical* code have produced medians of 2.66, 2.77, 2.91 and 3.00.

Anything below about 15% is therefore not resolvable end to end at batch 1,
which is what made the row-kernel A/B inconclusive until it was measured at
the kernel instead. Use `adi bench-linear` and the `non_expert` counter, not
`decode-batch`, to judge a change aimed at this path.

## How much headroom batch-1 decode has

Every packed weight is read once per token and no kernel change alters that,
so the first question is what the machine can stream. Measured on this
8-core EPYC 9645 VM with a STREAM-style probe, best of five, working sets far
larger than the 16 MB L3:

| | 1 thread | 8 threads |
| --- | ---: | ---: |
| sequential read, 2 GB | 41.2 GB/s | 58.9 GB/s |
| STREAM triad, 256 MB/array | 39.1 GB/s | 50.7 GB/s |
| sequential read, 8 MB (L3-resident) | 90.2 GB/s | 83.0 GB/s |

A decode step's own counters give the footprint. `adi decode-token` reports
`non_expert work_items=1405091840` and `expert work_items=1006632960`, so a
token reads 1.405e9 non-expert weights at 4 bits and 1.007e9 expert weights at
1.5 bits: **702.5 MB plus 188.7 MB, 891 MB total**. At the 0.362 s step
measured alongside those counters that is **2.46 GB/s**.

So batch-1 decode runs at about **4% of what the machine sustains**, and there
is roughly 24x of bandwidth headroom rather than the 5x that would still have
left the kernels responsible. Decode is not near a bandwidth floor; the
matvec is the constraint, and batching remains a throughput lever rather than
the only one.

The cache geometry matters for the sections below and `lscpu` reports it
misleadingly. Per `/sys/devices/system/cpu/cpu0/cache`: L1d is 64 KB private,
L2 is **512 KB private**, and L3 is **16 MB shared by all eight cores** --
not the "128 MiB (8 instances)" that `lscpu` prints. The 512 KB non-expert
state-value table is therefore exactly the size of one core's entire L2.

### The accumulator, not the lookup

The 512 KB state table does miss L1 on essentially every lookup. Counted with
`perf stat` over a standalone replica of the tile loop at the 8192x2048
in_proj_qkv shape, one thread, fixed setup cancelled by differencing two
round counts:

| inner loop | cycles/pass | instructions/pass | L1 misses/pass | IPC |
| --- | ---: | ---: | ---: | ---: |
| current kernel | 46.5 M | 180.2 M | 7.94 M | 3.87 |
| arithmetic decode into a 4 KB tlut | 60.1 M | 268.2 M | 0.14 M | 4.47 |
| row accumulator in a register | 31.2 M | 242.9 M | 7.94 M | 7.78 |
| row accumulator + signed 8 KB tlut | 31.7 M | 268.2 M | 0.14 M | 8.47 |

A pass decodes 8.39 M states, so the current kernel misses L1 on **95% of its
state lookups** and the arithmetic decode removes 98% of those misses. It is
still slower: the misses are L2 hits that the out-of-order engine already
overlaps, so they cost throughput the core had to spare, while the decode adds
88 M instructions per pass directly onto the serial chain from the state to
the address. Cycles rise 29%.

What the kernel actually spent its time on is the accumulator. State i lands
in local row `i >> 3`, so eight consecutive states -- sixteen adds -- all land
in the same `row_sums` slot, and because the row index moves under the flat
walk the compiler keeps that slot in memory. The sixteen adds then chain
through store-to-load forwarding rather than through the adder. Grouping the
walk by row lets the running sum stay in a register across the eight states
that touch it, in the same order, so the result is unchanged: 46.5 M cycles
to 31.2 M, with IPC going 3.87 to 7.78 on *more* instructions, which is the
signature of removing a stall rather than removing work.

The batch SIMD kernel already had this structure and says so in
`ne_accumulate_tile_row`; only the single-vector path was left on the flat
walk. `for_each_ne_row_state` in `ne_trellis.hpp` now gives both the same
grouping.

Two things were tried on top and are not worth keeping. Folding the sign into
a 1024-entry 8 KB table -- `(product >> 6) & 0x3FF` is exactly
`row | (negative << 9)`, so it is three integer ops and one L1-resident load
-- lands within 2% of the plain table (31.7 M against 31.2 M) despite 98%
fewer misses. Loading the value pair as one 8-byte load instead of two 4-byte
loads is likewise inside the noise. Splitting the row sum into two
accumulators, which would not have been bit-exact, was also no faster, so the
remaining chain is not the limit either.

### What the row accumulator is worth

Interleaved A/B of two binaries, alternating run by run so host drift cancels.
This VM's spread is wide -- the standalone replica shows run-to-run ranges
near 100% at eight threads -- so medians over many pairs, not single runs.

| | before | after | |
| --- | ---: | ---: | ---: |
| `bench-ne`, 8192x2048, 60 iterations, 8 pairs | 2.594 ms | 2.008 ms | 1.29x |
| `bench-linear` layer 0, `non_expert` total, 15 pairs | 6.524 ms | 5.141 ms | 1.27x |
| `decode-token`, `non_expert` total, 11 pairs | 213.1 ms | 172.6 ms | 1.23x |
| `decode-token`, step, 11 pairs | 0.3807 s | 0.3372 s | 1.13x |

The `bench-linear` distributions barely overlap: every "before" sample but one
falls between 6.43 and 6.98 ms, every "after" sample between 4.92 and 5.99.
The step gain follows from the kernel gain and the share it holds -- the
non-expert matvec is about 55% of a batch-1 step, so 1.23x on it predicts
1.12x overall against the 1.13x measured.

Decode is still nowhere near memory-bound after this: 702.5 MB of non-expert
weights in 172.6 ms is 4.07 GB/s, about 7% of the machine's 58.9 GB/s. The
kernel now runs at IPC near 8, so the next lever is instruction count rather
than stalls or bandwidth.

### Exactness

Bit-exact, not tolerance-bounded:

- the row-grouped walk `for_each_ne_row_state` reproduces the flat walk's
  state sequence, and its local row and step decomposition, on the same 4,000
  randomized tiles and single-bit edge cases the flat walk is checked against,
  including the transition that wraps the stream;
- the 64-token prefill keeps logits checksum `0x8c24a3f4644e03d3` and state
  checksum `0xc75f92166aad6875` through every change above;
- the 256-token prefill keeps `0x35a5f9bc4d1133d2` and `0xd3861b9b0d49fdd`,
  and the 1024-token prefill keeps `0xf53711345a669617` and
  `0x17a85ab867215ab`, at every microbatch in the sweeps;
- the SIMD batch kernel equals the scalar kernel at batches {1, 2, 3, 4, 5, 7,
  8, 15, 16, 17, 33, 64, 127} on three matrix shapes under both AVX2 and
  AVX-512, and each batch row equals the single-vector kernel;
- both ISA translation units compile with FP contraction disabled and contain
  no FMA instruction, so every lane performs the same scalar accumulation
  sequence, in the same order, as the scalar reference;
- the scalar loop is retained as `expert_accumulate_tiles_scalar` and runs on
  ISAs without a batch kernel;
- the full suite passes under `ADI_CPU_ISA` of scalar, avx2, and avx512.
