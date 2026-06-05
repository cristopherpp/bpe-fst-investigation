#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class MorphologicalFST {
public:
    struct State {
        std::size_t id{};
        std::string name;
        bool accepting{};
        std::vector<std::size_t> rule_indices;
    };

    struct Transition {
        std::size_t from{};
        std::size_t to{};
        std::string input;
        std::string output;
    };

    struct MorphologicalRule {
        std::string name;
        std::string suffix;
        std::string output;
        int priority{};
    };

    explicit MorphologicalFST(std::size_t min_root_symbols = 2);

    std::size_t add_state(std::string name = {}, bool accepting = false);

    void add_suffix_rule(std::string name,
                         std::string suffix,
                         std::string output = {},
                         int priority = 0);

    void add_root(std::string root);
    void add_roots(const std::vector<std::string>& roots);
    void clear_roots();

    void set_min_root_symbols(std::size_t value) noexcept;
    [[nodiscard]] std::size_t min_root_symbols() const noexcept;

    [[nodiscard]] std::vector<std::string> segment_word(std::string_view word) const;
    [[nodiscard]] std::vector<std::vector<std::string>>
    segment_corpus(const std::vector<std::string>& words) const;

    [[nodiscard]] const std::vector<State>& states() const noexcept;
    [[nodiscard]] const std::vector<Transition>& transitions() const noexcept;
    [[nodiscard]] const std::vector<MorphologicalRule>& rules() const noexcept;
    [[nodiscard]] std::size_t root_count() const noexcept;

    [[nodiscard]] std::size_t memory_estimate_bytes() const noexcept;


private:
    struct SymbolSpan {
        std::size_t begin{};
        std::size_t end{};
        std::string text;
    };

    struct SuffixMatch {
        std::size_t start_symbol{};
        std::size_t end_symbol{};
        std::size_t rule_index{};
    };

    struct Cell {
        bool valid{};
        std::vector<std::size_t> rule_indices;
        int priority{};
    };

    static constexpr std::size_t Npos = static_cast<std::size_t>(-1);

    [[nodiscard]] static std::vector<SymbolSpan>
    split_utf8_with_offsets(std::string_view text);

    [[nodiscard]] bool is_valid_root(std::string_view root,
                                     std::size_t root_symbols) const;

    [[nodiscard]] std::size_t find_transition(std::size_t from,
                                              std::string_view input) const;

    [[nodiscard]] std::vector<SuffixMatch>
    suffix_matches_ending_at(const std::vector<SymbolSpan>& symbols,
                             std::size_t end_symbol) const;

    [[nodiscard]] static bool better_cell(const Cell& candidate,
                                          const Cell& current);

    std::vector<State> states_;
    std::vector<Transition> transitions_;
    std::unordered_map<std::size_t, std::vector<std::size_t>> outgoing_;

    std::vector<MorphologicalRule> rules_;
    std::unordered_set<std::string> roots_;

    std::size_t min_root_symbols_;
};
