#pragma once

#include "BPETokenizer.hpp"
#include "MorphologicalFST.hpp"

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

class FSTBPETokenizer {
public:
    explicit FSTBPETokenizer(MorphologicalFST fst = MorphologicalFST{},
                             std::size_t max_merges = 100,
                             std::size_t min_pair_frequency = 2);

    void train(const std::vector<std::string>& corpus);
    void train(const std::vector<std::string>& corpus, std::size_t max_merges);

    [[nodiscard]] std::vector<std::string> tokenize_word(std::string_view word) const;
    [[nodiscard]] std::vector<std::string> tokenize(const std::vector<std::string>& words) const;

    [[nodiscard]] MorphologicalFST& fst() noexcept;
    [[nodiscard]] const MorphologicalFST& fst() const noexcept;

    [[nodiscard]] BPETokenizer& bpe() noexcept;
    [[nodiscard]] const BPETokenizer& bpe() const noexcept;

    [[nodiscard]] std::size_t vocabulary_size() const noexcept;
    [[nodiscard]] std::size_t memory_estimate_bytes() const noexcept;

private:
    MorphologicalFST fst_;
    BPETokenizer bpe_;
};
