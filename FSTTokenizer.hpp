#pragma once

#include "MorphologicalFST.hpp"

#include <cstddef>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

class FSTTokenizer {
public:
    explicit FSTTokenizer(MorphologicalFST fst = MorphologicalFST{});

    void train(const std::vector<std::string>& corpus);

    [[nodiscard]] std::vector<std::string> tokenize_word(std::string_view word) const;
    [[nodiscard]] std::vector<std::string> tokenize(const std::vector<std::string>& words) const;

    [[nodiscard]] MorphologicalFST& fst() noexcept;
    [[nodiscard]] const MorphologicalFST& fst() const noexcept;

    [[nodiscard]] std::size_t vocabulary_size() const noexcept;
    [[nodiscard]] std::size_t memory_estimate_bytes() const noexcept;

private:
    MorphologicalFST fst_;
    std::unordered_set<std::string> learned_vocabulary_;
};
