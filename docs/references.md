# Implementation references

ADI is an independent implementation. The following projects and model files
were used to understand formats and execution order:

- **Mach local MoE engine and Mach-1 checkpoint decoder** — the authoritative
  trellis, wave-gamma, SU/SV, embedding, and head codec contracts. The engine is
  Apache-2.0 licensed; the local checkpoint is also Apache-2.0 licensed.
- **llama.cpp (Bonsai fork)** — Qwen3.5/3.6 MoE tensor geometry, hybrid
  full-attention/Gated-DeltaNet execution order, GGUF conventions, and
  Responses API behavior. llama.cpp is MIT licensed.
- **DwarfStar (ds4)** — inspiration for a narrow, mmap-backed,
  model-specialized runtime with correctness-first CPU kernels. DwarfStar is
  MIT licensed.

No upstream source files are vendored. ADI's codec kernels, GGUF reader,
packager, model executor, tokenizer, JSON parser, and HTTP server were written
for this repository.
