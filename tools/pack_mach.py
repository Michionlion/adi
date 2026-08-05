#!/usr/bin/env python3
"""Re-containerize a native Mach-1 checkpoint as an ADI GGUF.

Tensor bytes are copied exactly. This tool never decodes or quantizes weights.
Safetensors dimensions are reversed in the GGUF directory because GGUF records
the contiguous dimension first.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ALIGNMENT = 32
QWEN35_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|"
    r"\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
GGUF_TYPES = {
    "F32": 0, "F16": 1, "I8": 24, "U8": 24, "I16": 25, "U16": 25,
    "I32": 26, "U32": 26, "I64": 27, "U64": 27, "F64": 28, "BF16": 30,
    "BOOL": 24,
}
DTYPE_BYTES = {
    "F32": 4, "F16": 2, "I8": 1, "U8": 1, "I16": 2, "U16": 2,
    "I32": 4, "U32": 4, "I64": 8, "U64": 8, "F64": 8, "BF16": 2,
    "BOOL": 1,
}


@dataclass(frozen=True)
class TensorSource:
    name: str
    path: Path
    source_offset: int
    length: int
    shape: tuple[int, ...]
    dtype: str
    gguf_offset: int = 0


def align_up(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def gguf_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def metadata_value(value: object) -> tuple[int, bytes]:
    if isinstance(value, bool):
        return 7, struct.pack("<?", value)
    if isinstance(value, str):
        return 8, gguf_string(value)
    if isinstance(value, int) and 0 <= value <= 0xFFFFFFFF:
        return 4, struct.pack("<I", value)
    if isinstance(value, int) and 0 <= value <= 0xFFFFFFFFFFFFFFFF:
        return 10, struct.pack("<Q", value)
    if isinstance(value, float):
        return 6, struct.pack("<f", value)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        encoded = bytearray(struct.pack("<IQ", 8, len(value)))
        for item in value:
            encoded.extend(gguf_string(item))
        return 9, bytes(encoded)
    if isinstance(value, list) and all(
            isinstance(item, int) and 0 <= item <= 0xFFFFFFFF for item in value):
        encoded = bytearray(struct.pack("<IQ", 4, len(value)))
        for item in value:
            encoded.extend(struct.pack("<I", item))
        return 9, bytes(encoded)
    raise TypeError(f"unsupported metadata value: {value!r}")


def read_safetensors(path: Path, prefix: str) -> list[TensorSource]:
    file_size = path.stat().st_size
    with path.open("rb") as source:
        raw_length = source.read(8)
        if len(raw_length) != 8:
            raise ValueError(f"{path}: truncated safetensors header")
        header_length = struct.unpack("<Q", raw_length)[0]
        if header_length > file_size - 8:
            raise ValueError(f"{path}: invalid safetensors header length")
        header = json.loads(source.read(header_length))

    data_start = 8 + header_length
    tensors: list[TensorSource] = []
    for key in sorted(key for key in header if key != "__metadata__"):
        descriptor = header[key]
        dtype = descriptor["dtype"]
        shape = tuple(int(extent) for extent in descriptor["shape"])
        begin, end = (int(offset) for offset in descriptor["data_offsets"])
        if dtype not in GGUF_TYPES:
            raise ValueError(f"{path}: unsupported dtype {dtype} for {key}")
        elements = 1
        for extent in shape:
            if extent <= 0:
                raise ValueError(f"{path}: invalid shape for {key}")
            elements *= extent
        expected = elements * DTYPE_BYTES[dtype]
        if end < begin or end - begin != expected:
            raise ValueError(f"{path}: byte count mismatch for {key}")
        if data_start + end > file_size:
            raise ValueError(f"{path}: tensor {key} points outside file")
        tensors.append(TensorSource(
            name=f"{prefix}{key}",
            path=path,
            source_offset=data_start + begin,
            length=end - begin,
            shape=shape,
            dtype=dtype,
        ))
    return tensors


def add_file(tensors: list[TensorSource], path: Path, prefix: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    tensors.extend(read_safetensors(path, prefix))


def collect_checkpoint(root: Path) -> list[TensorSource]:
    tensors: list[TensorSource] = []
    add_file(tensors, root / "extras.safetensors", "hf.")
    add_file(tensors, root / "packed/experts/codebook.safetensors", "mach.tlut.expert.")
    add_file(tensors, root / "packed/ne/tlut.safetensors", "mach.tlut.ne.")
    add_file(tensors, root / "packed/ne/embed_int4.safetensors", "mach.embedding.")
    for layer in range(40):
        add_file(tensors, root / f"packed/experts/L{layer:02d}.safetensors",
                 f"mach.expert.{layer}.")
        add_file(tensors, root / f"packed/ne/L{layer:02d}.safetensors",
                 f"mach.ne.{layer}.")
    for chunk in range(8):
        add_file(tensors, root / f"packed/head/head_c{chunk}of8.safetensors",
                 f"mach.output.{chunk}.")
    if len({tensor.name for tensor in tensors}) != len(tensors):
        raise ValueError("checkpoint produces duplicate GGUF tensor names")
    return tensors


def checkpoint_metadata(root: Path) -> dict[str, object]:
    config = json.loads((root / "config.json").read_text())
    text = config["text_config"]
    expected_layers = [
        "full_attention" if (layer + 1) % 4 == 0 else "linear_attention"
        for layer in range(40)
    ]
    if (
        config.get("model_type") != "qwen3_5_moe"
        or config.get("architectures") != ["Qwen3_5MoeForConditionalGeneration"]
        or text.get("model_type") != "qwen3_5_moe_text"
        or text.get("layer_types") != expected_layers
    ):
        raise ValueError("checkpoint is not the supported Qwen3.5-MoE architecture")
    rope = text["rope_parameters"]
    codec = json.loads((root / "packed/experts/codec.json").read_text())
    cb = codec["cb_params"]
    tokenizer = json.loads((root / "tokenizer.json").read_text())
    expected_pre_tokenizer = {
        "type": "Sequence",
        "pretokenizers": [
            {
                "type": "Split",
                "pattern": {"Regex": QWEN35_PATTERN},
                "behavior": "Isolated",
                "invert": False,
            },
            {
                "type": "ByteLevel",
                "add_prefix_space": False,
                "trim_offsets": False,
                "use_regex": False,
            },
        ],
    }
    expected_decoder = {
        "type": "ByteLevel",
        "add_prefix_space": False,
        "trim_offsets": False,
        "use_regex": False,
    }
    if (
        tokenizer.get("normalizer") != {"type": "NFC"}
        or tokenizer.get("pre_tokenizer") != expected_pre_tokenizer
        or tokenizer.get("decoder") != expected_decoder
        or {
            key: value
            for key, value in tokenizer.get("model", {}).items()
            if key not in {"vocab", "merges"}
        }
        != {
            "type": "BPE",
            "dropout": None,
            "unk_token": None,
            "continuing_subword_prefix": "",
            "end_of_word_suffix": "",
            "fuse_unk": False,
            "byte_fallback": False,
            "ignore_merges": False,
        }
        or any(
            added.get("single_word") is not False
            or added.get("lstrip") is not False
            or added.get("rstrip") is not False
            or added.get("normalized") is not False
            for added in tokenizer.get("added_tokens", [])
        )
    ):
        raise ValueError("checkpoint has an unsupported tokenizer pipeline")
    vocabulary_size = int(text["vocab_size"])
    tokens = [""] * vocabulary_size
    token_types = [5] * vocabulary_size
    for token, token_id in tokenizer["model"]["vocab"].items():
        tokens[int(token_id)] = token
        token_types[int(token_id)] = 1
    for added in tokenizer["added_tokens"]:
        token_id = int(added["id"])
        tokens[token_id] = added["content"]
        token_types[token_id] = 3 if added.get("special", False) else 4
    populated = [token_id for token_id, token in enumerate(tokens) if token]
    effective_vocabulary = max(populated) + 1
    if any(not tokens[token_id] for token_id in range(effective_vocabulary)):
        raise ValueError("tokenizer vocabulary has a non-tail ID gap")
    for token_id, token in enumerate(tokens):
        if not token:
            tokens[token_id] = f"[PAD{token_id}]"
    metadata = {
        "general.architecture": "qwen35moe",
        "general.name": "Mach-1-Additive-35B",
        "general.alignment": ALIGNMENT,
        "adi.format_version": 1,
        "adi.additive": True,
        "adi.model_family": "mach1",
        "adi.expert_codec": codec["container"],
        "adi.ne_codec": codec["ne_tier"]["codec"],
        "adi.embedding_codec": "affine_int4_g64_exceptions",
        "adi.head_codec": "int5_g64",
        "adi.regular_rms_norm": "qwen3next_zero_centered",
        "adi.gated_rms_norm": "multiplicative",
        "adi.tokenizer_normalizer": "nfc_unicode_15_0",
        "adi.tokenizer_pretokenizer": "qwen35_unicode_16_0_regex",
        "adi.expert.k_num": 3,
        "adi.expert.k_den": 2,
        "adi.expert.l": int(cb["L"]),
        "adi.expert.v": int(cb["V"]),
        "adi.expert.tlut_bits": int(cb["tlut_bits"]),
        "adi.expert.tile_x": int(cb["td_x"]),
        "adi.expert.tile_y": int(cb["td_y"]),
        "qwen35moe.context_length": int(text["max_position_embeddings"]),
        "qwen35moe.embedding_length": int(text["hidden_size"]),
        "qwen35moe.block_count": int(text["num_hidden_layers"]),
        "qwen35moe.expert_feed_forward_length": int(text["moe_intermediate_size"]),
        "qwen35moe.expert_shared_feed_forward_length":
            int(text["shared_expert_intermediate_size"]),
        "qwen35moe.expert_count": int(text["num_experts"]),
        "qwen35moe.expert_used_count": int(text["num_experts_per_tok"]),
        "qwen35moe.attention.head_count": int(text["num_attention_heads"]),
        "qwen35moe.attention.head_count_kv": int(text["num_key_value_heads"]),
        "qwen35moe.attention.layer_norm_rms_epsilon": float(text["rms_norm_eps"]),
        "qwen35moe.rope.freq_base": float(rope["rope_theta"]),
        "qwen35moe.full_attention_interval": int(text["full_attention_interval"]),
        "qwen35moe.ssm.conv_kernel": int(text["linear_conv_kernel_dim"]),
        "qwen35moe.ssm.inner_size":
            int(text["linear_num_value_heads"] * text["linear_value_head_dim"]),
        "qwen35moe.ssm.state_size": int(text["linear_value_head_dim"]),
        "qwen35moe.ssm.time_step_rank": int(text["linear_num_value_heads"]),
        "qwen35moe.ssm.group_count": int(text["linear_num_key_heads"]),
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "qwen2",
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.token_type": token_types,
        "tokenizer.ggml.merges": tokenizer["model"]["merges"],
        "tokenizer.ggml.bos_token_id": int(text["bos_token_id"]),
        "tokenizer.ggml.eos_token_id": int(text["eos_token_id"]),
        "tokenizer.chat_template": (root / "chat_template.jinja").read_text(),
    }
    return metadata


def assign_offsets(tensors: Iterable[TensorSource]) -> list[TensorSource]:
    result = []
    offset = 0
    for tensor in tensors:
        offset = align_up(offset)
        result.append(TensorSource(
            tensor.name, tensor.path, tensor.source_offset, tensor.length,
            tensor.shape, tensor.dtype, offset,
        ))
        offset += tensor.length
    return result


def build_header(metadata: dict[str, object],
                 tensors: list[TensorSource]) -> tuple[bytes, int]:
    body = bytearray(b"GGUF")
    body.extend(struct.pack("<IQQ", 3, len(tensors), len(metadata)))
    for key, value in metadata.items():
        value_type, encoded = metadata_value(value)
        body.extend(gguf_string(key))
        body.extend(struct.pack("<I", value_type))
        body.extend(encoded)
    for tensor in tensors:
        body.extend(gguf_string(tensor.name))
        body.extend(struct.pack("<I", len(tensor.shape)))
        for extent in reversed(tensor.shape):
            body.extend(struct.pack("<Q", extent))
        body.extend(struct.pack("<IQ", GGUF_TYPES[tensor.dtype], tensor.gguf_offset))
    data_offset = align_up(len(body))
    body.extend(b"\0" * (data_offset - len(body)))
    return bytes(body), data_offset


def copy_range(source_path: Path, offset: int, length: int, output) -> None:
    with source_path.open("rb") as source:
        source.seek(offset)
        remaining = length
        while remaining:
            chunk = source.read(min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise IOError(f"{source_path}: unexpected EOF while copying tensor")
            output.write(chunk)
            remaining -= len(chunk)


def write_gguf(destination: Path, metadata: dict[str, object],
               input_tensors: list[TensorSource]) -> tuple[int, int]:
    tensors = assign_offsets(input_tensors)
    header, data_offset = build_header(metadata, tensors)
    payload_size = 0 if not tensors else tensors[-1].gguf_offset + tensors[-1].length
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    try:
        with temporary.open("wb") as output:
            output.write(header)
            position = 0
            for tensor in tensors:
                padding = tensor.gguf_offset - position
                if padding:
                    output.write(b"\0" * padding)
                copy_range(tensor.path, tensor.source_offset, tensor.length, output)
                position = tensor.gguf_offset + tensor.length
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return data_offset + payload_size, len(tensors)


def format_size(size: int) -> str:
    return f"{size / (1024 ** 3):.2f} GiB"


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="adi-pack-test-") as directory:
        root = Path(directory)
        source = root / "source.safetensors"
        values = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
        descriptor = {
            "weight": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]}
        }
        header = json.dumps(descriptor, separators=(",", ":")).encode()
        header += b" " * ((8 - len(header) % 8) % 8)
        source.write_bytes(struct.pack("<Q", len(header)) + header + values)
        tensors = read_safetensors(source, "test.")
        output = root / "test.gguf"
        size, count = write_gguf(
            output,
            {"general.architecture": "qwen35moe", "adi.additive": True},
            tensors,
        )
        assert count == 1
        assert output.stat().st_size == size
        assert output.read_bytes()[-16:] == values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", nargs="?", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--plan", action="store_true",
                        help="validate and report without writing")
    parser.add_argument("--self-test", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.checkpoint is None:
        print("pack_mach.py: checkpoint is required", file=sys.stderr)
        return 2
    root = args.checkpoint.resolve()
    tensors = collect_checkpoint(root)
    metadata = checkpoint_metadata(root)
    positioned = assign_offsets(tensors)
    header, data_offset = build_header(metadata, positioned)
    payload = 0 if not positioned else positioned[-1].gguf_offset + positioned[-1].length
    total = data_offset + payload
    print(f"source: {root}")
    print(f"tensors: {len(tensors)}")
    print(f"metadata: {len(metadata)}")
    print(f"output size: {format_size(total)}")
    if args.plan:
        return 0
    if args.output is None:
        print("pack_mach.py: --output is required unless --plan is used",
              file=sys.stderr)
        return 2
    size, count = write_gguf(args.output.resolve(), metadata, tensors)
    print(f"wrote: {args.output.resolve()} ({format_size(size)}, {count} tensors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
