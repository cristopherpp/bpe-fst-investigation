#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

class BPETokenizer {
public:
    struct MergeRule {
        std::string left;
        std::string right;
        std::string merged;
        std::size_t rank{};
        std::size_t frequency{};
    };

    explicit BPETokenizer(std::size_t max_merges = 100,
                          std::size_t min_pair_frequency = 2);

    void train(const std::vector<std::string>& corpus);
    void train(const std::vector<std::string>& corpus, std::size_t max_merges);

    [[nodiscard]] std::vector<std::string> tokenize_word(std::string_view word) const;
    [[nodiscard]] std::vector<std::string> tokenize(const std::vector<std::string>& words) const;

    void save_merges(const std::string& path) const;
    void load_merges(const std::string& path);

    [[nodiscard]] const std::vector<MergeRule>& merges() const noexcept;
    [[nodiscard]] std::size_t vocabulary_size() const noexcept;
    [[nodiscard]] std::size_t memory_estimate_bytes() const noexcept;

    [[nodiscard]] std::size_t max_merges() const noexcept;
    void set_max_merges(std::size_t value) noexcept;

    [[nodiscard]] std::size_t min_pair_frequency() const noexcept;
    void set_min_pair_frequency(std::size_t value) noexcept;

    [[nodiscard]] static std::vector<std::string> split_utf8_symbols(std::string_view text);

private:
    struct WordEntry {
        std::vector<std::string> symbols;
        std::size_t frequency{};
    };

    struct PairCount {
        std::string left;
        std::string right;
        std::size_t count{};
    };

    static constexpr std::string_view EndOfWord = "</w>";

    [[nodiscard]] static std::string pair_key(std::string_view left, std::string_view right);
    [[nodiscard]] static bool ends_with(std::string_view value, std::string_view suffix) noexcept;

    static void merge_pair_in_symbols(std::vector<std::string>& symbols,
                                      std::string_view left,
                                      std::string_view right,
                                      std::string_view merged);

    [[nodiscard]] static std::vector<std::string>
    strip_end_of_word(const std::vector<std::string>& symbols);

    void rebuild_merge_index();
    void rebuild_vocabulary(const std::vector<WordEntry>& entries);

    std::size_t max_merges_;
    std::size_t min_pair_frequency_;

    std::vector<MergeRule> merges_;
    std::unordered_map<std::string, std::size_t> merge_rank_;
    std::unordered_set<std::string> learned_vocabulary_;
};
