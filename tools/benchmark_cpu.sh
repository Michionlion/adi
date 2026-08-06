#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "usage: tools/benchmark_cpu.sh MODEL.gguf [--full]" >&2
    exit 2
fi

model_path=$1
full=${2:-}
if [[ -n "$full" && "$full" != "--full" ]]; then
    echo "second argument must be --full" >&2
    exit 2
fi

binary=./build/adi
if [[ ! -x "$binary" ]]; then
    echo "build/adi is missing; build the Release configuration first" >&2
    exit 1
fi

echo "CPU"
lscpu | sed -n \
    -e '/^Architecture:/p' \
    -e '/^CPU(s):/p' \
    -e '/^Model name:/p' \
    -e '/^Thread(s) per core:/p'
echo "Compiler"
c++ --version | sed -n '1p'
echo "ADI_THREADS: ${ADI_THREADS:-auto}"
echo "ADI_CPU_ISA: ${ADI_CPU_ISA:-auto}"
echo

"$binary" validate "$model_path"
echo
"$binary" bench-expert "$model_path" 0 0 gate 10
echo
"$binary" bench-ne "$model_path" 0 \
    model.language_model.layers.0.mlp.shared_expert.gate_proj.weight 10
echo
"$binary" bench-ne "$model_path" 0 \
    model.language_model.layers.0.mlp.shared_expert.gate_proj.weight 10 4
echo
"$binary" bench-moe "$model_path" 0 5
echo
"$binary" bench-attention "$model_path" 3 5
echo
"$binary" bench-linear "$model_path" 0 5
echo
"$binary" bench-head "$model_path" 0 5
echo
"$binary" bench-head "$model_path" 0 5 4

if [[ "$full" == "--full" ]]; then
    echo
    "$binary" decode-token "$model_path" 248044
fi
