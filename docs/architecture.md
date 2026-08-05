# Architecture

ADI has one vertical execution path:

```text
Responses request
  -> prompt rendering and BPE
  -> Qwen3.6 model loop
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
- `Model` owns validated Qwen configuration, tensor views, and KV state.
- `Scratch` owns request-local temporary buffers.
- The backend operation table contains stateless kernel entry points.

Model data is immutable after load. A server process currently owns one model
and serializes inference. Request parsing may be concurrent later, but batching
is outside the first implementation.

## Backend seam

The model calls a compact operation table for dense GEMV, additive GEMV,
embedding lookup, output projection, and RMS normalization. Attention, routing,
sampling, and the layer loop remain one shared model path. The CPU table is the
only implementation initially. A later CUDA table must consume the same tensor
views and produce the same results; it must not fork model execution.

## Deliberate omissions

- No generic architecture registry.
- No training, conversion during load, or runtime quantization.
- No chat-completions, completions, embeddings, reranking, or admin API.
- No multimodal path.
- No speculative decoding, batching, disk KV cache, or distributed execution.
- No graph compiler or general tensor library.
