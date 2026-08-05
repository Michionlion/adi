# ADI contributor notes

ADI is a narrow inference runtime for additive-weight models. Mach-1-Additive
is the reference model. It is not a general GGUF runner.

## Design rules

- Keep one model path and one public API.
- Keep packed weights mmap-backed. Never create a full dense-weight shadow.
- Preserve the packed Mach-1 arithmetic. Do not requantize it at load time.
- Make CPU correctness obvious before optimizing it.
- Add a backend operation only when the model needs it.
- Keep model semantics outside backend kernels so a later CUDA backend shares
  the same execution code.
- Prefer contiguous, immutable model data and explicit scratch arenas.
- Reject unsupported metadata, tensor types, shapes, and codecs immediately.
- Avoid feature flags that create permanent semantic variants.
- Expose only the OpenAI-compatible Responses API.

## Code rules

- C++20, standard library first.
- Keep public headers small and ownership explicit.
- Use exceptions only during setup. Return status at request-time boundaries.
- Comment non-obvious tensor shapes, layout, arithmetic, and cache lifetimes.
- Do not add a dependency when a small, well-tested implementation is clearer.
- Do not commit model files, build output, or benchmark output.

## Validation

- Every codec needs a golden-vector test against the released Python decoder.
- Every model stage needs shape checks and a deterministic reference test.
- Run `cmake --build build` and `ctest --test-dir build` before each commit.
- Do not claim tokens/second until end-to-end token generation is correct.

## Git

- Commit one coherent feature at a time.
- Do not push unless the user asks explicitly.
