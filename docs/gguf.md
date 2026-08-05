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
| `adi.regular_rms_norm` | string | `qwen3next_zero_centered` |
| `adi.gated_rms_norm` | string | `multiplicative` |
| `adi.tokenizer_normalizer` | string | `nfc_unicode_15_0` |
| `adi.tokenizer_pretokenizer` | string | `qwen35_unicode_16_0_regex` |

Qwen architecture and tokenizer metadata use the established llama.cpp GGUF
keys where they apply. ADI deliberately reads only the values required by its
single supported model architecture.

Format version 1 copies `extras.safetensors` unchanged. Regular Qwen3Next
RMSNorm weights—including decoder input/post-attention norms, the final norm,
and full-attention Q/K norms—are zero-centered and execute with a `1 + weight`
multiplier. Gated DeltaNet `RMSNormGated` weights are conventional
multiplicative weights and execute without the offset. Original version-1
files predate the explicit metadata keys above but have the same required
semantics.

The tokenizer contract is the checkpoint's Unicode 15.0 NFC normalizer
followed by its Unicode 16.0 Qwen3.5 regex split and byte-level BPE. The
versions differ because that is the pipeline implemented by the checkpoint's
`tokenizers` release. The packer rejects a
checkpoint whose `tokenizer.json` pipeline differs from that contract.
The checkpoint's `tokenizer.chat_template` is also required to match the
supported Qwen3.5 template fingerprint. ADI implements that template's
text-only, thinking-enabled specialization and rejects other templates.

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

The packager consolidates the checkpoint's files into one GGUF while retaining
its 32-expert chunks for direct indexed access. Names use these stable forms:

- `mach.expert.{layer}.e{chunk_base}.{gate|up|down}.{trellis|su|sv|wave_gamma}`
- `mach.ne.{layer}.{original_tensor_name}|{trellis|SU|SV|Wscale}`
- `mach.embedding.{q_packed|mn|mx|exc_idx|exc_bits}`
- `mach.output.{chunk}.{source_key}`
- `mach.tlut.{expert|ne}.tlut`

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
