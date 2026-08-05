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
- **Unicode Character Database 15.0 and 16.0** — versioned NFC, general
  category, whitespace, and case-folding data used to reproduce the
  checkpoint tokenizer. The generated tables are derived from Unicode data
  files under the [Unicode Data Files and Software
  License](https://www.unicode.org/license.txt).

No upstream source files are vendored. The generated Unicode tables are
checked in so the runtime build remains self-contained. ADI's codec kernels,
GGUF reader, packager, model executor, tokenizer logic, JSON parser, and HTTP
server were written for this repository.
