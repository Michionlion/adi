# Architecture

ADI has one vertical execution path:

```text
Responses request
  -> prompt rendering and BPE
  -> Qwen3.5-MoE model loop
  -> backend operation table
  -> CPU additive and dense kernels
  -> sampler
  -> Responses events or JSON
```

## Model storage

The model is one mmap-backed GGUF v3 file. Standard GGUF scalar and tensor
types carry every byte. `adi.*` metadata describes how packed tensors combine
into a logical additive matrix.

Large expert shards are consolidated while packaging:

- one trellis tensor per layer and projection;
- one `su`, `sv`, and `wave_gamma` tensor beside it;
- no per-expert allocation or tensor object at runtime.

Non-expert trellis tensors, the int4 embedding, int5 output head, and retained
BF16 scalars remain separate logical matrices because their codecs differ.

Unknown architecture, codec version, shape, or required tensor is a load error.

## Runtime ownership

- `MappedFile` owns the read-only model mapping.
- `GgufFile` owns parsed metadata and tensor descriptors pointing into it.
- `Model` owns validated Qwen configuration and immutable cached descriptors.
- `DecoderState` owns request-local KV, convolution, and recurrent state.
- `Scratch` owns request-local temporary buffers.
- The backend operation table contains runtime-dispatched kernel entry points.

Model data is immutable after load. Construction resolves all layer, expert,
attention, norm, embedding, and output-head views once. RoPE powers are cached
with the model, while each request computes a position's sin/cos pairs once.

The executor advances independent sequences and prompt tokens layer-major.
Non-expert, BF16, grouped expert, and output projections accept batches so a
packed weight stream can serve multiple vectors. The server's continuous
batcher admits new requests between decode steps and advances active requests
together. A persistent process-wide worker pool executes row and expert tasks;
`ADI_THREADS` selects its size.

## Backend seam

The model calls a compact operation table for dense GEMV, additive GEMV,
embedding lookup, output projection, and RMS normalization. Attention, routing,
sampling, and the layer loop remain one shared model path. The CPU table is the
only implementation initially. It selects scalar, AVX2, AVX-512, NEON, or
SVE-capable helpers at runtime. `ADI_CPU_ISA=scalar` retains the reference path
for verification. VNNI integer dot products are not used for float activations
because doing so would require runtime activation quantization and change model
arithmetic. A later CUDA table must consume the same tensor views and produce
the same results; it must not fork model execution.

## Deliberate omissions

- No generic architecture registry.
- No training, conversion during load, or runtime quantization.
- No chat-completions, completions, embeddings, reranking, or admin API.
- No multimodal path.
- No speculative decoding, disk KV cache, or distributed execution.
- No graph compiler or general tensor library.
