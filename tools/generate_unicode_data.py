#!/usr/bin/env python3
"""Generate compact Unicode tables for ADI's checkpoint tokenizer.

The checkpoint pipeline uses Unicode 15.0 NFC normalization and Unicode 16.0
regex properties. The checked-in output keeps the C++ runtime self-contained.
By default this script downloads both versioned Unicode Character Databases
from unicode.org; directory options support offline regeneration.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path


MAX_CODEPOINT = 0x110000
HANGUL_BEGIN = 0xAC00
HANGUL_END = 0xD7A4
NORMALIZATION_VERSION = "15.0.0"
REGEX_VERSION = "16.0.0"
REGEX_FILES = (
    "UnicodeData.txt",
    "CaseFolding.txt",
    "PropList.txt",
)
NORMALIZATION_FILES = (
    "UnicodeData.txt",
    "DerivedNormalizationProps.txt",
)
# tokenizers 0.22.2's NFC table intentionally serves as the checkpoint
# reference. It differs from UCD 15.0 for this one canonical composition.
TOKENIZERS_NFC_EXCLUSIONS = {0x11938}


def codepoint_range(value: str) -> range:
    bounds = value.strip().split("..")
    first = int(bounds[0], 16)
    last = int(bounds[-1], 16)
    return range(first, last + 1)


def read_ucd(
    directory: Path | None,
    version: str,
    filenames: tuple[str, ...],
) -> dict[str, str]:
    if directory is not None:
        return {
            name: (directory / name).read_text(encoding="utf-8")
            for name in filenames
        }
    base = f"https://www.unicode.org/Public/{version}/ucd"
    return {
        name: urllib.request.urlopen(f"{base}/{name}").read().decode("utf-8")
        for name in filenames
    }


def grouped_ranges(values: list[int]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    previous = None
    for codepoint, value in enumerate(values):
        if value != previous:
            ranges.append((codepoint, value))
            previous = value
    return ranges


def canonical_order(
    codepoints: list[int], combining: list[int]
) -> list[int]:
    result = list(codepoints)
    for index in range(1, len(result)):
        current_class = combining[result[index]]
        if current_class == 0:
            continue
        position = index
        while position > 0:
            previous_class = combining[result[position - 1]]
            if previous_class == 0 or previous_class <= current_class:
                break
            result[position], result[position - 1] = (
                result[position - 1],
                result[position],
            )
            position -= 1
    return result


def parse_unicode_data(text: str):
    categories = ["Cn"] * MAX_CODEPOINT
    combining = [0] * MAX_CODEPOINT
    raw_decomposition: dict[int, list[int]] = {}
    pending_range: tuple[int, str, int] | None = None

    for raw_line in text.splitlines():
        fields = raw_line.split(";")
        codepoint = int(fields[0], 16)
        name = fields[1]
        category = fields[2]
        combining_class = int(fields[3])
        if name.endswith(", First>"):
            pending_range = (codepoint, category, combining_class)
            continue
        if name.endswith(", Last>"):
            if pending_range is None:
                raise ValueError("UnicodeData Last entry has no First entry")
            first, first_category, first_combining = pending_range
            if (category, combining_class) != (
                first_category,
                first_combining,
            ):
                raise ValueError("UnicodeData range properties differ")
            for value in range(first, codepoint + 1):
                categories[value] = category
                combining[value] = combining_class
            pending_range = None
            continue
        categories[codepoint] = category
        combining[codepoint] = combining_class
        decomposition = fields[5]
        if decomposition and not decomposition.startswith("<"):
            raw_decomposition[codepoint] = [
                int(value, 16) for value in decomposition.split()
            ]
    if pending_range is not None:
        raise ValueError("UnicodeData First entry has no Last entry")
    return categories, combining, raw_decomposition


def parse_data(
    regex_data: dict[str, str],
    normalization_data: dict[str, str],
):
    regex_categories, _, _ = parse_unicode_data(regex_data["UnicodeData.txt"])
    _, combining, raw_decomposition = parse_unicode_data(
        normalization_data["UnicodeData.txt"]
    )

    whitespace: set[int] = set()
    for raw_line in regex_data["PropList.txt"].splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        codepoints, property_name = (
            value.strip() for value in line.split(";", 1)
        )
        if property_name == "White_Space":
            whitespace.update(codepoint_range(codepoints))

    casefold: list[tuple[int, int]] = []
    for raw_line in regex_data["CaseFolding.txt"].splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        codepoint, status, mapping = (
            value.strip() for value in line.split(";", 3)[:3]
        )
        values = mapping.split()
        if status in {"C", "S"} and len(values) == 1:
            casefold.append((int(codepoint, 16), int(values[0], 16)))

    composition_exclusions: set[int] = set()
    for raw_line in normalization_data[
        "DerivedNormalizationProps.txt"
    ].splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        fields = [value.strip() for value in line.split(";")]
        if len(fields) >= 2 and fields[1] == "Full_Composition_Exclusion":
            composition_exclusions.update(codepoint_range(fields[0]))
    composition_exclusions.update(TOKENIZERS_NFC_EXCLUSIONS)

    properties = [
        (1 if category.startswith("L") else 0)
        | (2 if category.startswith("N") else 0)
        | (4 if category.startswith("M") else 0)
        | (8 if codepoint in whitespace else 0)
        for codepoint, category in enumerate(regex_categories)
    ]

    def fully_decompose(codepoint: int) -> list[int]:
        if codepoint not in raw_decomposition:
            return [codepoint]
        result = []
        for value in raw_decomposition[codepoint]:
            result.extend(fully_decompose(value))
        return canonical_order(result, combining)

    decompositions = []
    decomposition_values = []
    for codepoint in sorted(raw_decomposition):
        if HANGUL_BEGIN <= codepoint < HANGUL_END:
            continue
        values = fully_decompose(codepoint)
        offset = len(decomposition_values)
        decomposition_values.extend(values)
        decompositions.append((codepoint, offset, len(values)))

    compositions = []
    for codepoint, values in raw_decomposition.items():
        if len(values) == 2 and codepoint not in composition_exclusions:
            compositions.append((*values, codepoint))

    return (
        properties,
        combining,
        sorted(casefold),
        decompositions,
        decomposition_values,
        sorted(compositions),
    )


def render(
    regex_data: dict[str, str],
    normalization_data: dict[str, str],
) -> str:
    (
        properties,
        combining,
        casefold,
        decompositions,
        decomposition_values,
        compositions,
    ) = parse_data(regex_data, normalization_data)
    lines = [
        "// Generated by tools/generate_unicode_data.py with "
        f"Unicode {NORMALIZATION_VERSION} NFC and Unicode {REGEX_VERSION} "
        "regex properties.",
        "constexpr std::array property_ranges{",
    ]
    lines.extend(
        f"    PropertyRange{{0x{codepoint:06X}U, {value}U}},"
        for codepoint, value in grouped_ranges(properties)
    )
    lines.append("};")
    lines.append("constexpr std::array combining_ranges{")
    lines.extend(
        f"    CombiningRange{{0x{codepoint:06X}U, {value}U}},"
        for codepoint, value in grouped_ranges(combining)
    )
    lines.append("};")
    lines.append("constexpr std::array casefold_entries{")
    lines.extend(
        f"    CasefoldEntry{{0x{source:06X}U, 0x{target:06X}U}},"
        for source, target in casefold
    )
    lines.append("};")
    lines.append("constexpr std::array decomposition_values{")
    for begin in range(0, len(decomposition_values), 8):
        values = decomposition_values[begin : begin + 8]
        lines.append(
            "    " + ", ".join(f"0x{value:06X}U" for value in values) + ","
        )
    lines.append("};")
    lines.append("constexpr std::array decompositions{")
    lines.extend(
        f"    DecompositionEntry{{0x{codepoint:06X}U, {offset}U, {length}U}},"
        for codepoint, offset, length in decompositions
    )
    lines.append("};")
    lines.append("constexpr std::array compositions{")
    lines.extend(
        f"    CompositionEntry{{0x{first:06X}U, 0x{second:06X}U, "
        f"0x{composed:06X}U}},"
        for first, second, composed in compositions
    )
    lines.append("};")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output", nargs="?", default="src/unicode_data.inc", type=Path
    )
    parser.add_argument(
        "--regex-ucd-dir",
        type=Path,
        help="directory containing the required Unicode 16 UCD text files",
    )
    parser.add_argument(
        "--normalization-ucd-dir",
        type=Path,
        help="directory containing the required Unicode 15 UCD text files",
    )
    args = parser.parse_args()
    regex_data = read_ucd(
        args.regex_ucd_dir, REGEX_VERSION, REGEX_FILES
    )
    normalization_data = read_ucd(
        args.normalization_ucd_dir,
        NORMALIZATION_VERSION,
        NORMALIZATION_FILES,
    )
    args.output.write_text(render(regex_data, normalization_data))


if __name__ == "__main__":
    main()
