#include "unicode.hpp"

#include "utf8.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>

namespace adi {
namespace {

struct PropertyRange {
    std::uint32_t first;
    std::uint8_t flags;
};

struct CombiningRange {
    std::uint32_t first;
    std::uint8_t combining_class;
};

struct CasefoldEntry {
    std::uint32_t source;
    std::uint32_t target;
};

struct DecompositionEntry {
    std::uint32_t codepoint;
    std::uint16_t offset;
    std::uint8_t length;
};

struct CompositionEntry {
    std::uint32_t first;
    std::uint32_t second;
    std::uint32_t composed;
};

#include "unicode_data.inc"

template <typename Range, typename Member>
const Range &range_for(
    const auto &ranges,
    std::uint32_t codepoint,
    Member member) noexcept {
    const auto found = std::upper_bound(
        ranges.begin(), ranges.end(), codepoint,
        [&](std::uint32_t value, const Range &range) {
            return value < range.*member;
        });
    return *(found - 1);
}

std::uint8_t combining_class(std::uint32_t codepoint) noexcept {
    return range_for<CombiningRange>(
               combining_ranges, codepoint, &CombiningRange::first)
        .combining_class;
}

std::uint32_t compose_pair(
    std::uint32_t first,
    std::uint32_t second) noexcept {
    constexpr std::uint32_t s_base = 0xAC00;
    constexpr std::uint32_t l_base = 0x1100;
    constexpr std::uint32_t v_base = 0x1161;
    constexpr std::uint32_t t_base = 0x11A7;
    constexpr std::uint32_t l_count = 19;
    constexpr std::uint32_t v_count = 21;
    constexpr std::uint32_t t_count = 28;
    constexpr std::uint32_t n_count = v_count * t_count;
    constexpr std::uint32_t s_count = l_count * n_count;

    if (first >= l_base && first < l_base + l_count &&
        second >= v_base && second < v_base + v_count) {
        return s_base + (first - l_base) * n_count +
               (second - v_base) * t_count;
    }
    if (first >= s_base && first < s_base + s_count &&
        (first - s_base) % t_count == 0 &&
        second > t_base && second < t_base + t_count) {
        return first + second - t_base;
    }

    const auto found = std::lower_bound(
        compositions.begin(), compositions.end(),
        std::pair{first, second},
        [](const CompositionEntry &entry,
           const std::pair<std::uint32_t, std::uint32_t> &key) {
            return std::pair{entry.first, entry.second} < key;
        });
    if (found != compositions.end() && found->first == first &&
        found->second == second) {
        return found->composed;
    }
    return 0;
}

void append_decomposed(
    std::uint32_t codepoint,
    std::vector<std::uint32_t> &output) {
    constexpr std::uint32_t s_base = 0xAC00;
    constexpr std::uint32_t l_base = 0x1100;
    constexpr std::uint32_t v_base = 0x1161;
    constexpr std::uint32_t t_base = 0x11A7;
    constexpr std::uint32_t v_count = 21;
    constexpr std::uint32_t t_count = 28;
    constexpr std::uint32_t n_count = v_count * t_count;
    constexpr std::uint32_t s_count = 19 * n_count;
    if (codepoint >= s_base && codepoint < s_base + s_count) {
        const auto index = codepoint - s_base;
        output.push_back(l_base + index / n_count);
        output.push_back(v_base + (index % n_count) / t_count);
        if (index % t_count != 0) {
            output.push_back(t_base + index % t_count);
        }
        return;
    }

    const auto found = std::lower_bound(
        decompositions.begin(), decompositions.end(), codepoint,
        [](const DecompositionEntry &entry, std::uint32_t value) {
            return entry.codepoint < value;
        });
    if (found == decompositions.end() || found->codepoint != codepoint) {
        output.push_back(codepoint);
        return;
    }
    output.insert(
        output.end(),
        decomposition_values.begin() + found->offset,
        decomposition_values.begin() + found->offset + found->length);
}

bool contraction_char(std::uint32_t codepoint, char expected) noexcept {
    return unicode_casefold(codepoint) ==
           static_cast<std::uint32_t>(expected);
}

} // namespace

UnicodeProperties unicode_properties(std::uint32_t codepoint) noexcept {
    if (codepoint > 0x10FFFFU) {
        return {};
    }
    const auto flags =
        range_for<PropertyRange>(
            property_ranges, codepoint, &PropertyRange::first)
            .flags;
    return {
        (flags & 1U) != 0,
        (flags & 2U) != 0,
        (flags & 4U) != 0,
        (flags & 8U) != 0,
    };
}

std::uint32_t unicode_casefold(std::uint32_t codepoint) noexcept {
    const auto found = std::lower_bound(
        casefold_entries.begin(), casefold_entries.end(), codepoint,
        [](const CasefoldEntry &entry, std::uint32_t value) {
            return entry.source < value;
        });
    return found != casefold_entries.end() && found->source == codepoint
               ? found->target
               : codepoint;
}

std::vector<std::uint32_t> utf8_codepoints(std::string_view text) {
    if (!valid_utf8(text)) {
        throw std::runtime_error("invalid UTF-8");
    }
    std::vector<std::uint32_t> result;
    result.reserve(text.size());
    for (std::size_t position = 0; position < text.size();) {
        const auto first = static_cast<unsigned char>(text[position++]);
        if (first <= 0x7FU) {
            result.push_back(first);
            continue;
        }
        std::size_t continuation_count = 0;
        std::uint32_t codepoint = 0;
        if (first <= 0xDFU) {
            continuation_count = 1;
            codepoint = first & 0x1FU;
        } else if (first <= 0xEFU) {
            continuation_count = 2;
            codepoint = first & 0x0FU;
        } else {
            continuation_count = 3;
            codepoint = first & 0x07U;
        }
        for (std::size_t index = 0; index < continuation_count; ++index) {
            codepoint = (codepoint << 6) |
                        (static_cast<unsigned char>(text[position++]) & 0x3FU);
        }
        result.push_back(codepoint);
    }
    return result;
}

std::string codepoints_utf8(const std::vector<std::uint32_t> &codepoints) {
    std::string result;
    for (const auto codepoint : codepoints) {
        append_utf8(result, codepoint);
    }
    return result;
}

std::string normalize_nfc(std::string_view text) {
    const auto source = utf8_codepoints(text);
    std::vector<std::uint32_t> decomposed;
    decomposed.reserve(source.size());
    for (const auto codepoint : source) {
        append_decomposed(codepoint, decomposed);
    }

    // Canonical ordering is a stable insertion sort within each starter segment.
    for (std::size_t index = 1; index < decomposed.size(); ++index) {
        const auto current_class = combining_class(decomposed[index]);
        if (current_class == 0) {
            continue;
        }
        auto position = index;
        while (position > 0) {
            const auto previous_class = combining_class(decomposed[position - 1]);
            if (previous_class == 0 || previous_class <= current_class) {
                break;
            }
            std::swap(decomposed[position], decomposed[position - 1]);
            --position;
        }
    }
    if (decomposed.empty()) {
        return {};
    }

    std::vector<std::uint32_t> composed;
    composed.reserve(decomposed.size());
    composed.push_back(decomposed[0]);
    std::size_t starter_position = 0;
    std::uint32_t starter = decomposed[0];
    std::uint8_t previous_class = 0;
    for (std::size_t index = 1; index < decomposed.size(); ++index) {
        const auto codepoint = decomposed[index];
        const auto current_class = combining_class(codepoint);
        const auto combined =
            (previous_class == 0 || previous_class < current_class)
                ? compose_pair(starter, codepoint)
                : 0;
        if (combined != 0) {
            composed[starter_position] = combined;
            starter = combined;
            continue;
        }
        if (current_class == 0) {
            starter_position = composed.size();
            starter = codepoint;
        }
        composed.push_back(codepoint);
        previous_class = current_class;
    }
    return codepoints_utf8(composed);
}

std::vector<std::string> qwen35_pretokenize(
    std::string_view normalized_text) {
    const auto codepoints = utf8_codepoints(normalized_text);
    std::vector<std::string> result;
    const auto properties = [&](std::size_t position) {
        return position < codepoints.size()
                   ? unicode_properties(codepoints[position])
                   : UnicodeProperties{};
    };
    const auto add = [&](std::size_t begin, std::size_t end) {
        std::vector<std::uint32_t> piece(
            codepoints.begin() + static_cast<std::ptrdiff_t>(begin),
            codepoints.begin() + static_cast<std::ptrdiff_t>(end));
        result.push_back(codepoints_utf8(piece));
    };

    for (std::size_t position = 0; position < codepoints.size();) {
        const auto begin = position;
        const auto codepoint = codepoints[position];
        const auto flags = properties(position);

        if (codepoint == '\'' && position + 1 < codepoints.size()) {
            if (contraction_char(codepoints[position + 1], 's') ||
                contraction_char(codepoints[position + 1], 't') ||
                contraction_char(codepoints[position + 1], 'm') ||
                contraction_char(codepoints[position + 1], 'd')) {
                position += 2;
                add(begin, position);
                continue;
            }
            if (position + 2 < codepoints.size() &&
                ((contraction_char(codepoints[position + 1], 'r') &&
                  contraction_char(codepoints[position + 2], 'e')) ||
                 (contraction_char(codepoints[position + 1], 'v') &&
                  contraction_char(codepoints[position + 2], 'e')) ||
                 (contraction_char(codepoints[position + 1], 'l') &&
                  contraction_char(codepoints[position + 2], 'l')))) {
                position += 3;
                add(begin, position);
                continue;
            }
        }

        const bool optional_letter_prefix =
            codepoint != '\r' && codepoint != '\n' && !flags.letter &&
            !flags.number && position + 1 < codepoints.size() &&
            (properties(position + 1).letter || properties(position + 1).mark);
        if (flags.letter || flags.mark || optional_letter_prefix) {
            if (optional_letter_prefix) {
                ++position;
            }
            while (position < codepoints.size()) {
                const auto current = properties(position);
                if (!current.letter && !current.mark) {
                    break;
                }
                ++position;
            }
            add(begin, position);
            continue;
        }

        if (flags.number) {
            add(begin, ++position);
            continue;
        }

        auto punctuation_position =
            codepoint == ' ' ? position + 1 : position;
        if (punctuation_position < codepoints.size()) {
            const auto first = properties(punctuation_position);
            if (!first.whitespace && !first.letter && !first.mark &&
                !first.number) {
                position = punctuation_position;
                while (position < codepoints.size()) {
                    const auto current = properties(position);
                    if (current.whitespace || current.letter || current.mark ||
                        current.number) {
                        break;
                    }
                    ++position;
                }
                while (position < codepoints.size() &&
                       (codepoints[position] == '\r' ||
                        codepoints[position] == '\n')) {
                    ++position;
                }
                add(begin, position);
                continue;
            }
        }

        std::size_t whitespace_count = 0;
        std::size_t last_newline_end = 0;
        while (position + whitespace_count < codepoints.size() &&
               properties(position + whitespace_count).whitespace) {
            const auto current = codepoints[position + whitespace_count];
            if (current == '\r' || current == '\n') {
                last_newline_end = position + whitespace_count + 1;
            }
            ++whitespace_count;
        }
        if (last_newline_end != 0) {
            position = last_newline_end;
            add(begin, position);
            continue;
        }
        if (whitespace_count > 1 &&
            position + whitespace_count < codepoints.size()) {
            position += whitespace_count - 1;
            add(begin, position);
            continue;
        }
        if (whitespace_count != 0) {
            position += whitespace_count;
            add(begin, position);
            continue;
        }

        add(begin, ++position);
    }
    return result;
}

} // namespace adi
