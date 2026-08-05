#include "bpe.hpp"

#include <cassert>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

std::vector<std::uint32_t> reference_merge(
    std::vector<std::uint32_t> symbols,
    const adi::BpeRuleMap &rules) {
    while (symbols.size() > 1) {
        auto best_rank = std::numeric_limits<std::uint32_t>::max();
        std::uint64_t best_key = 0;
        std::uint32_t replacement = 0;
        for (std::size_t index = 0; index + 1 < symbols.size(); ++index) {
            const auto key = adi::bpe_pair_key(symbols[index], symbols[index + 1]);
            const auto found = rules.find(key);
            if (found != rules.end() && found->second.first < best_rank) {
                best_rank = found->second.first;
                best_key = key;
                replacement = found->second.second;
            }
        }
        if (best_rank == std::numeric_limits<std::uint32_t>::max()) {
            break;
        }
        std::vector<std::uint32_t> merged;
        for (std::size_t index = 0; index < symbols.size();) {
            if (index + 1 < symbols.size() &&
                adi::bpe_pair_key(symbols[index], symbols[index + 1]) == best_key) {
                merged.push_back(replacement);
                index += 2;
            } else {
                merged.push_back(symbols[index++]);
            }
        }
        symbols = std::move(merged);
    }
    return symbols;
}

void verify_sequences(
    std::vector<std::uint32_t> &sequence,
    std::size_t remaining,
    const adi::BpeRuleMap &rules) {
    if (remaining == 0) {
        assert(adi::bpe_merge(sequence, rules) == reference_merge(sequence, rules));
        return;
    }
    for (std::uint32_t symbol = 0; symbol < 4; ++symbol) {
        sequence.push_back(symbol);
        verify_sequences(sequence, remaining - 1, rules);
        sequence.pop_back();
    }
}

} // namespace

int main() {
    const adi::BpeRuleMap rules{
        {adi::bpe_pair_key(0, 1), {0, 4}},
        {adi::bpe_pair_key(1, 2), {1, 5}},
        {adi::bpe_pair_key(4, 2), {2, 6}},
        {adi::bpe_pair_key(0, 0), {3, 7}},
        {adi::bpe_pair_key(7, 7), {4, 8}},
        {adi::bpe_pair_key(4, 4), {5, 9}},
        {adi::bpe_pair_key(9, 2), {6, 10}},
    };

    std::vector<std::uint32_t> sequence;
    for (std::size_t length = 0; length <= 8; ++length) {
        verify_sequences(sequence, length, rules);
    }

    bool cancelled = false;
    try {
        const std::vector<std::uint32_t> input(4096, 0);
        (void)adi::bpe_merge(input, rules, [] { return true; });
    } catch (const std::runtime_error &) {
        cancelled = true;
    }
    assert(cancelled);
}
