#include "FSTTokenizer.hpp"

#include <iterator>
#include <utility>

FSTTokenizer::FSTTokenizer(MorphologicalFST fst)
    : fst_(std::move(fst)) {}

void FSTTokenizer::train(const std::vector<std::string>& corpus) {
    learned_vocabulary_.clear();

    // First pass: use a rootless copy to infer likely roots from the training
    // corpus. This avoids the first learned root making later root discovery too
    // strict. Rules are configured by language, while roots are induced from data.
    auto rootless_fst = fst_;
    rootless_fst.clear_roots();

    for (const auto& word : corpus) {
        if (word.empty()) {
            continue;
        }

        const auto segments = rootless_fst.segment_word(word);

        if (!segments.empty()) {
            fst_.add_root(segments.front());
        } else {
            fst_.add_root(word);
        }
    }


    // Second pass: rebuild the learned vocabulary after root induction. The
    // vocabulary is the set of morpheme-like FST output symbols observed during
    // training.
    for (const auto& word : corpus) {
        auto segments = tokenize_word(word);

        for (auto& segment : segments) {
            if (!segment.empty()) {
                learned_vocabulary_.insert(std::move(segment));
            }
        }
    }
}

std::vector<std::string> FSTTokenizer::tokenize_word(std::string_view word) const {
    return fst_.segment_word(word);
}

std::vector<std::string> FSTTokenizer::tokenize(const std::vector<std::string>& words) const {
    std::vector<std::string> output;

    for (const auto& word : words) {
        auto tokens = tokenize_word(word);
        output.insert(output.end(),
                      std::make_move_iterator(tokens.begin()),
                      std::make_move_iterator(tokens.end()));
    }

    return output;
}

MorphologicalFST& FSTTokenizer::fst() noexcept {
    return fst_;
}

const MorphologicalFST& FSTTokenizer::fst() const noexcept {
    return fst_;
}

std::size_t FSTTokenizer::vocabulary_size() const noexcept {
    return learned_vocabulary_.size();
}

std::size_t FSTTokenizer::memory_estimate_bytes() const noexcept {
    std::size_t bytes = sizeof(*this) + fst_.memory_estimate_bytes();

    bytes += learned_vocabulary_.bucket_count() * sizeof(void*);

    for (const auto& token : learned_vocabulary_) {
        bytes += sizeof(token) + token.capacity();
    }

    return bytes;
}
