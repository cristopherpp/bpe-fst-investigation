#include "FSTBPETokenizer.hpp"

#include <utility>

FSTBPETokenizer::FSTBPETokenizer(MorphologicalFST fst,
                                 std::size_t max_merges,
                                 std::size_t min_pair_frequency)
    : fst_(std::move(fst)),
      bpe_(max_merges, min_pair_frequency) {}

void FSTBPETokenizer::train(const std::vector<std::string>& corpus) {
    std::vector<std::string> segmented_corpus;

    for (const auto& word : corpus) {
        auto segments = fst_.segment_word(word);

        for (auto& segment : segments) {
            if (!segment.empty()) {
                segmented_corpus.push_back(std::move(segment));
            }
        }
    }

    bpe_.train(segmented_corpus);
}

void FSTBPETokenizer::train(const std::vector<std::string>& corpus,
                            std::size_t max_merges) {
    bpe_.set_max_merges(max_merges);
    train(corpus);
}

std::vector<std::string> FSTBPETokenizer::tokenize_word(std::string_view word) const {
    const auto segments = fst_.segment_word(word);

    std::vector<std::string> output;

    for (const auto& segment : segments) {
        auto tokens = bpe_.tokenize_word(segment);
        output.insert(output.end(),
                      std::make_move_iterator(tokens.begin()),
                      std::make_move_iterator(tokens.end()));
    }

    return output;
}

std::vector<std::string> FSTBPETokenizer::tokenize(const std::vector<std::string>& words) const {
    std::vector<std::string> output;

    for (const auto& word : words) {
        auto tokens = tokenize_word(word);
        output.insert(output.end(),
                      std::make_move_iterator(tokens.begin()),
                      std::make_move_iterator(tokens.end()));
    }

    return output;
}

MorphologicalFST& FSTBPETokenizer::fst() noexcept {
    return fst_;
}

const MorphologicalFST& FSTBPETokenizer::fst() const noexcept {
    return fst_;
}

BPETokenizer& FSTBPETokenizer::bpe() noexcept {
    return bpe_;
}

const BPETokenizer& FSTBPETokenizer::bpe() const noexcept {
    return bpe_;
}

std::size_t FSTBPETokenizer::vocabulary_size() const noexcept {
    return bpe_.vocabulary_size();
}

std::size_t FSTBPETokenizer::memory_estimate_bytes() const noexcept {
    return sizeof(*this) + fst_.memory_estimate_bytes() + bpe_.memory_estimate_bytes();
}
