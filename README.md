# ADI

ADI is a small C++ inference runtime built only for additive-weight language
models. Its flagship target is Mach-1-Additive-35B.

The runtime keeps Mach-1's trellis codes and scale surfaces packed in memory
and evaluates them directly. It does not reconstruct or requantize the full
model. Model files use GGUF as a container with `adi.*` metadata and standard
integer/float tensor payload types.

Initial scope:

- Linux x86-64 CPU inference.
- Qwen3.6-35B-A3B (`qwen3_5_moe`) as used by Mach-1.
- One OpenAI-compatible endpoint: `POST /v1/responses`.
- Text generation only.

CUDA is a later backend, not part of the initial implementation.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build
```

Validate a native Mach-1 checkpoint and show the planned GGUF size:

```bash
python3 tools/pack_mach.py /path/to/Mach-1-Additive-35B --plan
```

Package it without decoding or requantizing any weights:

```bash
python3 tools/pack_mach.py /path/to/Mach-1-Additive-35B \
  --output models/Mach-1-Additive-35B.gguf
./build/adi inspect models/Mach-1-Additive-35B.gguf
```

Start the Responses API:

```bash
./build/adi serve --model models/Mach-1-Additive-35B.gguf \
  --host 127.0.0.1 --port 8080
```

See [`docs/api.md`](docs/api.md) for the request shape and a curl example.

The current CPU measurements and reproduction command are in
[`docs/benchmarks.md`](docs/benchmarks.md). Implementation references are
recorded in [`docs/references.md`](docs/references.md).

## Repository layout

- `include/adi/`: public runtime API.
- `src/`: loader, model, kernels, tokenizer, and server.
- `tests/`: unit and golden-vector tests.
- `tools/`: model packaging and inspection tools.
- `docs/`: format and architecture contracts.
