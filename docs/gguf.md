# Additive GGUF profile

ADI uses GGUF v3 as a memory-mappable container, not as a signal to run generic
GGML quantization kernels. The Mach codec remains intact. Its streams and
parameters are stored as ordinary integer and floating-point tensors so other
GGUF readers can determine every tensor's size and safely skip it.

## Required metadata

| Key | GGUF type | Initial value |
| --- | --- | --- |
| `general.architecture` | string | `qwen35moe` |
| `adi.format_version` | uint32 | `1` |
| `adi.additive` | bool | `true` |
| `adi.model_family` | string | `mach1` |
| `adi.expert_codec` | string | `trained_susv_wave_gamma_chunked_v1` |
| `adi.ne_codec` | string | `canon_rht_bitshift_trellis_intlattice` |
| `adi.embedding_codec` | string | `affine_int4_g64_exceptions` |
| `adi.head_codec` | string | `int5_g64` |

Qwen architecture and tokenizer metadata use the established llama.cpp GGUF
keys where they apply. ADI deliberately reads only the values required by its
single supported model architecture.

The initial Mach-1 expert codec is identified by these values:

| Key | Type | Value |
| --- | --- | --- |
| `adi.expert.k_num` | uint32 | `3` |
| `adi.expert.k_den` | uint32 | `2` |
| `adi.expert.l` | uint32 | `16` |
| `adi.expert.v` | uint32 | `8` |
| `adi.expert.tlut_bits` | uint32 | `15` |
| `adi.expert.tile_x` | uint32 | `16` |
| `adi.expert.tile_y` | uint32 | `16` |

The rational `k_num/k_den` representation avoids making a floating-point
metadata value part of the bitstream contract.

## Tensor representation

Tensor dimensions follow GGUF order: the first dimension is contiguous.
Packed bytes use GGML `I8`; code words use `I16`; signs use `I8`; indexes use
`I32`; continuous scales and lookup tables use `F16`. Dense residual tensors
are `BF16` or `F32`. Signed GGML storage types are byte containers where the
codec defines an unsigned interpretation.

The packager consolidates the checkpoint's per-expert and per-head shards.
Names use the following stable prefixes:

- `mach.expert.{layer}.{gate|up|down}.{trellis|su|sv|wave_gamma}`
- `mach.ne.{original_tensor_name}.{trellis|su|sv|wscale}`
- `mach.embedding.{q_packed|mn|mx|exception_index|exception_bits}`
- `mach.output.{q_packed|group_scale|protected_rows|protected_dense}`
- `mach.tlut.{expert|ne}`

Non-expert tensors retain llama.cpp's Qwen3.5 MoE tensor names, such as
`blk.0.attn_norm.weight`. This keeps the architecture legible while the `mach.*`
prefix makes every codec-specific tensor impossible to mistake for a dense
weight.

## Execution contract

Expert reconstruction order is:

1. trellis states to integer-lattice tiles;
2. FP16 boundary cast;
3. one `wave_gamma` scale per 16x16 tile;
4. right Hadamard transform and `su` scaling;
5. left Hadamard transform and `sv` scaling;
6. crop from power-of-two padding.

The CPU kernel may fuse these stages with matrix-vector multiplication, but it
must preserve the reference operation order closely enough to pass the golden
codec fixtures. It must never materialize a complete dense expert bank.
