#include "MorphologicalFST.hpp"

#include <limits>
#include <stdexcept>
#include <utility>

namespace {
std::size_t utf8_char_length(unsigned char lead) {
    if ((lead & 0x80U) == 0U) {
        return 1;
    }

    if ((lead & 0xE0U) == 0xC0U) {
        return 2;
    }

    if ((lead & 0xF0U) == 0xE0U) {
        return 3;
    }

    if ((lead & 0xF8U) == 0xF0U) {
        return 4;
    }

    return 1;
}
}

MorphologicalFST::MorphologicalFST(std::size_t min_root_symbols)
    : min_root_symbols_(min_root_symbols) {
    add_state("q0", false);
}

std::size_t MorphologicalFST::add_state(std::string name, bool accepting) {
    const std::size_t id = states_.size();

    if (name.empty()) {
        name = "q" + std::to_string(id);
    }

    states_.push_back(State{
        .id = id,
        .name = std::move(name),
        .accepting = accepting,
        .rule_indices = {}
    });

    return id;
}

void MorphologicalFST::add_suffix_rule(std::string name,
                                       std::string suffix,
                                       std::string output,
                                       int priority) {
    if (suffix.empty()) {
        throw std::invalid_argument("Suffix rule cannot have an empty suffix.");
    }

    if (states_.empty()) {
        add_state("q0", false);
    }

    if (output.empty()) {
        output = suffix;
    }

    const std::size_t rule_index = rules_.size();

    rules_.push_back(MorphologicalRule{
        .name = std::move(name),
        .suffix = suffix,
        .output = std::move(output),
        .priority = priority
    });

    const auto symbols = split_utf8_with_offsets(suffix);

    std::size_t current = 0;

    for (auto it = symbols.rbegin(); it != symbols.rend(); ++it) {
        const std::size_t transition_index = find_transition(current, it->text);

        if (transition_index == Npos) {
            const std::size_t next = add_state();

            transitions_.push_back(Transition{
                .from = current,
                .to = next,
                .input = it->text,
                .output = {}
            });

            outgoing_[current].push_back(transitions_.size() - 1);
            current = next;
        } else {
            current = transitions_[transition_index].to;
        }
    }

    states_[current].accepting = true;
    states_[current].rule_indices.push_back(rule_index);
}

void MorphologicalFST::add_root(std::string root) {
    if (!root.empty()) {
        roots_.insert(std::move(root));
    }
}

void MorphologicalFST::add_roots(const std::vector<std::string>& roots) {
    for (const auto& root : roots) {
        add_root(root);
    }
}

void MorphologicalFST::clear_roots() {
    roots_.clear();
}

void MorphologicalFST::set_min_root_symbols(std::size_t value) noexcept {
    min_root_symbols_ = value;
}

std::size_t MorphologicalFST::min_root_symbols() const noexcept {
    return min_root_symbols_;
}

std::vector<std::string> MorphologicalFST::segment_word(std::string_view word) const {
    if (word.empty()) {
        return {};
    }

    const auto symbols = split_utf8_with_offsets(word);

    if (symbols.empty()) {
        return {std::string{word}};
    }

    const std::size_t n = symbols.size();

    std::vector<Cell> dp(n + 1);
    dp[n].valid = true;

    for (std::size_t end = n; end > 0; --end) {
        if (!dp[end].valid) {
            continue;
        }

        const auto matches = suffix_matches_ending_at(symbols, end);

        for (const auto& match : matches) {
            Cell candidate;
            candidate.valid = true;
            candidate.priority = rules_[match.rule_index].priority + dp[end].priority;
            candidate.rule_indices.reserve(1 + dp[end].rule_indices.size());
            candidate.rule_indices.push_back(match.rule_index);
            candidate.rule_indices.insert(candidate.rule_indices.end(),
                                          dp[end].rule_indices.begin(),
                                          dp[end].rule_indices.end());

            if (better_cell(candidate, dp[match.start_symbol])) {
                dp[match.start_symbol] = std::move(candidate);
            }
        }
    }

    bool found = false;
    std::size_t best_root_end = n;
    std::size_t best_covered = 0;
    int best_priority = std::numeric_limits<int>::min();
    std::size_t best_rule_count = 0;

    for (std::size_t root_end = 1; root_end < n; ++root_end) {
        if (!dp[root_end].valid || dp[root_end].rule_indices.empty()) {
            continue;
        }

        const std::size_t prefix_end_byte = symbols[root_end].begin;
        const std::string_view root{word.data(), prefix_end_byte};

        if (!is_valid_root(root, root_end)) {
            continue;
        }

        const std::size_t covered = n - root_end;
        const int priority = dp[root_end].priority;
        const std::size_t rule_count = dp[root_end].rule_indices.size();

        if (!found ||
            covered > best_covered ||
            (covered == best_covered && priority > best_priority) ||
            (covered == best_covered &&
             priority == best_priority &&
             rule_count > best_rule_count)) {
            found = true;
            best_root_end = root_end;
            best_covered = covered;
            best_priority = priority;
            best_rule_count = rule_count;
        }
    }

    if (!found) {
        return {std::string{word}};
    }

    const std::size_t prefix_end_byte = symbols[best_root_end].begin;

    std::vector<std::string> output;
    output.reserve(1 + dp[best_root_end].rule_indices.size());

    output.push_back(std::string{word.substr(0, prefix_end_byte)});

    for (const auto rule_index : dp[best_root_end].rule_indices) {
        output.push_back(rules_[rule_index].output);
    }

    return output;
}

std::vector<std::vector<std::string>>
MorphologicalFST::segment_corpus(const std::vector<std::string>& words) const {
    std::vector<std::vector<std::string>> output;
    output.reserve(words.size());

    for (const auto& word : words) {
        output.push_back(segment_word(word));
    }

    return output;
}

const std::vector<MorphologicalFST::State>&
MorphologicalFST::states() const noexcept {
    return states_;
}

const std::vector<MorphologicalFST::Transition>&
MorphologicalFST::transitions() const noexcept {
    return transitions_;
}

const std::vector<MorphologicalFST::MorphologicalRule>&
MorphologicalFST::rules() const noexcept {
    return rules_;
}

std::size_t MorphologicalFST::memory_estimate_bytes() const noexcept {
    std::size_t bytes = sizeof(*this);

    bytes += states_.capacity() * sizeof(State);
    for (const auto& state : states_) {
        bytes += state.name.capacity();
        bytes += state.rule_indices.capacity() * sizeof(std::size_t);
    }

    bytes += transitions_.capacity() * sizeof(Transition);
    for (const auto& transition : transitions_) {
        bytes += transition.input.capacity();
        bytes += transition.output.capacity();
    }

    bytes += outgoing_.bucket_count() * sizeof(void*);
    for (const auto& [state, edges] : outgoing_) {
        bytes += sizeof(state);
        bytes += edges.capacity() * sizeof(std::size_t);
    }

    bytes += rules_.capacity() * sizeof(MorphologicalRule);
    for (const auto& rule : rules_) {
        bytes += rule.name.capacity();
        bytes += rule.suffix.capacity();
        bytes += rule.output.capacity();
    }

    bytes += roots_.bucket_count() * sizeof(void*);
    for (const auto& root : roots_) {
        bytes += sizeof(root) + root.capacity();
    }

    return bytes;
}

std::vector<MorphologicalFST::SymbolSpan>
MorphologicalFST::split_utf8_with_offsets(std::string_view text) {
    std::vector<SymbolSpan> output;

    for (std::size_t i = 0; i < text.size();) {
        const auto lead = static_cast<unsigned char>(text[i]);
        std::size_t length = utf8_char_length(lead);

        if (i + length > text.size()) {
            length = 1;
        }

        output.push_back(SymbolSpan{
            .begin = i,
            .end = i + length,
            .text = std::string{text.substr(i, length)}
        });

        i += length;
    }

    return output;
}

bool MorphologicalFST::is_valid_root(std::string_view root,
                                     std::size_t root_symbols) const {
    if (root.empty()) {
        return false;
    }

    if (root_symbols < min_root_symbols_) {
        return false;
    }

    if (roots_.empty()) {
        return true;
    }

    return roots_.contains(std::string{root});
}

std::size_t MorphologicalFST::find_transition(std::size_t from,
                                              std::string_view input) const {
    const auto it = outgoing_.find(from);

    if (it == outgoing_.end()) {
        return Npos;
    }

    for (const auto transition_index : it->second) {
        const auto& transition = transitions_[transition_index];

        if (std::string_view{transition.input} == input) {
            return transition_index;
        }
    }

    return Npos;
}

std::vector<MorphologicalFST::SuffixMatch>
MorphologicalFST::suffix_matches_ending_at(const std::vector<SymbolSpan>& symbols,
                                           std::size_t end_symbol) const {
    std::vector<SuffixMatch> matches;

    if (states_.empty()) {
        return matches;
    }

    std::size_t state = 0;

    for (std::size_t position = end_symbol; position > 0; --position) {
        const auto& symbol = symbols[position - 1].text;
        const std::size_t transition_index = find_transition(state, symbol);

        if (transition_index == Npos) {
            break;
        }

        state = transitions_[transition_index].to;

        if (states_[state].accepting) {
            for (const auto rule_index : states_[state].rule_indices) {
                matches.push_back(SuffixMatch{
                    .start_symbol = position - 1,
                    .end_symbol = end_symbol,
                    .rule_index = rule_index
                });
            }
        }
    }

    return matches;
}

bool MorphologicalFST::better_cell(const Cell& candidate,
                                   const Cell& current) {
    if (!candidate.valid) {
        return false;
    }

    if (!current.valid) {
        return true;
    }

    if (candidate.priority != current.priority) {
        return candidate.priority > current.priority;
    }

    if (candidate.rule_indices.size() != current.rule_indices.size()) {
        return candidate.rule_indices.size() < current.rule_indices.size();
    }

    return false;
}
