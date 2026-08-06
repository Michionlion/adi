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
