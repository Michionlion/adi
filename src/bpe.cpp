#include "bpe.hpp"

#include <cstddef>
#include <limits>
#include <queue>
#include <set>
#include <stdexcept>
#include <unordered_map>

namespace adi {
namespace {

constexpr std::size_t no_node = std::numeric_limits<std::size_t>::max();

struct Node {
    std::uint32_t symbol;
    std::size_t previous = no_node;
    std::size_t next = no_node;
    bool active = true;
};

struct Candidate {
    std::uint32_t rank;
    std::uint64_t key;
};

struct CandidateGreater {
    bool operator()(const Candidate &left, const Candidate &right) const noexcept {
        if (left.rank != right.rank) {
            return left.rank > right.rank;
        }
        return left.key > right.key;
    }
};

void check_cancelled(
    const std::function<bool()> &cancelled,
    std::size_t &operations) {
    if ((operations++ & 1023U) == 0 && cancelled && cancelled()) {
        throw std::runtime_error("tokenization cancelled");
    }
}

} // namespace

std::vector<std::uint32_t> bpe_merge(
    std::span<const std::uint32_t> symbols,
    const BpeRuleMap &rules,
    const std::function<bool()> &cancelled) {
    if (symbols.size() < 2 || rules.empty()) {
        if (cancelled && cancelled()) {
            throw std::runtime_error("tokenization cancelled");
        }
        return {symbols.begin(), symbols.end()};
    }

    std::vector<Node> nodes;
    nodes.reserve(symbols.size());
    for (std::size_t index = 0; index < symbols.size(); ++index) {
        nodes.push_back({
            symbols[index],
            index == 0 ? no_node : index - 1,
            index + 1 == symbols.size() ? no_node : index + 1,
            true,
        });
    }

    std::unordered_map<std::uint64_t, std::set<std::size_t>> occurrences;
    std::priority_queue<
        Candidate,
        std::vector<Candidate>,
        CandidateGreater> candidates;
    std::size_t operations = 0;

    const auto remove_edge = [&](std::size_t left) {
        if (left == no_node || !nodes[left].active || nodes[left].next == no_node) {
            return;
        }
        const auto right = nodes[left].next;
        const auto key = bpe_pair_key(nodes[left].symbol, nodes[right].symbol);
        const auto found = occurrences.find(key);
        if (found == occurrences.end()) {
            return;
        }
        found->second.erase(left);
        if (found->second.empty()) {
            occurrences.erase(found);
        }
    };
    const auto add_edge = [&](std::size_t left) {
        if (left == no_node || !nodes[left].active || nodes[left].next == no_node) {
            return;
        }
        const auto right = nodes[left].next;
        const auto key = bpe_pair_key(nodes[left].symbol, nodes[right].symbol);
        const auto rule = rules.find(key);
        if (rule == rules.end()) {
            return;
        }
        auto &positions = occurrences[key];
        const bool was_empty = positions.empty();
        if (positions.insert(left).second && was_empty) {
            candidates.push({rule->second.first, key});
        }
    };

    for (std::size_t index = 0; index + 1 < nodes.size(); ++index) {
        check_cancelled(cancelled, operations);
        add_edge(index);
    }

    while (!candidates.empty()) {
        check_cancelled(cancelled, operations);
        const auto candidate = candidates.top();
        candidates.pop();
        const auto occurrence = occurrences.find(candidate.key);
        const auto rule = rules.find(candidate.key);
        if (occurrence == occurrences.end() || rule == rules.end() ||
            rule->second.first != candidate.rank) {
            continue;
        }

        auto positions = std::move(occurrence->second);
        occurrences.erase(occurrence);
        for (const auto left : positions) {
            check_cancelled(cancelled, operations);
            if (!nodes[left].active || nodes[left].next == no_node) {
                continue;
            }
            const auto right = nodes[left].next;
            if (bpe_pair_key(nodes[left].symbol, nodes[right].symbol) !=
                candidate.key) {
                continue;
            }

            const auto previous = nodes[left].previous;
            const auto next = nodes[right].next;
            remove_edge(previous);
            remove_edge(left);
            remove_edge(right);

            nodes[left].symbol = rule->second.second;
            nodes[left].next = next;
            nodes[right].active = false;
            nodes[right].previous = no_node;
            nodes[right].next = no_node;
            if (next != no_node) {
                nodes[next].previous = left;
            }

            add_edge(previous);
            add_edge(left);
        }
    }

    std::vector<std::uint32_t> result;
    result.reserve(symbols.size());
    for (auto node = std::size_t{0}; node != no_node; node = nodes[node].next) {
        result.push_back(nodes[node].symbol);
    }
    return result;
}

} // namespace adi
