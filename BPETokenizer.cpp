#include "BPETokenizer.hpp"

#include <fstream>
#include <ranges>
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

BPETokenizer::BPETokenizer(std::size_t max_merges,
                           std::size_t min_pair_frequency)
    : max_merges_(max_merges),
      min_pair_frequency_(min_pair_frequency) {}

void BPETokenizer::train(const std::vector<std::string>& corpus) {
    merges_.clear();
    merge_rank_.clear();
    learned_vocabulary_.clear();

    std::unordered_map<std::string, std::size_t> word_frequency;

    for (const auto& word : corpus | std::views::filter([](const std::string& value) {
             return !value.empty();
         })) {
        ++word_frequency[word];
    }

    std::vector<WordEntry> entries;
    entries.reserve(word_frequency.size());

    for (const auto& [word, frequency] : word_frequency) {
        auto symbols = split_utf8_symbols(word);
        symbols.emplace_back(EndOfWord);
        entries.push_back(WordEntry{std::move(symbols), frequency});
    }

    for (std::size_t iteration = 0; iteration < max_merges_; ++iteration) {
        std::unordered_map<std::string, PairCount> pair_counts;

        for (const auto& entry : entries) {
            if (entry.symbols.size() < 2) {
                continue;
            }

            for (std::size_t i = 0; i + 1 < entry.symbols.size(); ++i) {
                const auto& left = entry.symbols[i];
                const auto& right = entry.symbols[i + 1];

                auto key = pair_key(left, right);
                auto [it, inserted] = pair_counts.try_emplace(
                    std::move(key),
                    PairCount{left, right, 0}
                );

                it->second.count += entry.frequency;
            }
        }

        if (pair_counts.empty()) {
            break;
        }

        auto best = pair_counts.cend();

        for (auto it = pair_counts.cbegin(); it != pair_counts.cend(); ++it) {
            if (best == pair_counts.cend()) {
                best = it;
                continue;
            }

            const auto& current = it->second;
            const auto& selected = best->second;

            if (current.count > selected.count ||
                (current.count == selected.count && it->first < best->first)) {
                best = it;
            }
        }

        if (best == pair_counts.cend() ||
            best->second.count < min_pair_frequency_) {
            break;
        }

        MergeRule rule;
        rule.left = best->second.left;
        rule.right = best->second.right;
        rule.merged = rule.left + rule.right;
        rule.rank = merges_.size();
        rule.frequency = best->second.count;

        merges_.push_back(std::move(rule));

        const auto& stored_rule = merges_.back();
        merge_rank_[pair_key(stored_rule.left, stored_rule.right)] = stored_rule.rank;

        for (auto& entry : entries) {
            merge_pair_in_symbols(entry.symbols,
                                  stored_rule.left,
                                  stored_rule.right,
                                  stored_rule.merged);
        }
    }

    rebuild_vocabulary(entries);
}

void BPETokenizer::train(const std::vector<std::string>& corpus,
                         std::size_t max_merges) {
    max_merges_ = max_merges;
    train(corpus);
}

std::vector<std::string> BPETokenizer::tokenize_word(std::string_view word) const {
    auto symbols = split_utf8_symbols(word);

    if (symbols.empty()) {
        return {};
    }

    symbols.emplace_back(EndOfWord);

    for (const auto& rule : merges_) {
        merge_pair_in_symbols(symbols, rule.left, rule.right, rule.merged);
    }

    return strip_end_of_word(symbols);
}

std::vector<std::string> BPETokenizer::tokenize(const std::vector<std::string>& words) const {
    std::vector<std::string> output;

    for (const auto& word : words) {
        auto tokens = tokenize_word(word);
        output.insert(output.end(),
                      std::make_move_iterator(tokens.begin()),
                      std::make_move_iterator(tokens.end()));
    }

    return output;
}

void BPETokenizer::save_merges(const std::string& path) const {
    std::ofstream out(path);

    if (!out) {
        throw std::runtime_error("Could not open merge file for writing: " + path);
    }

    out << "# BPE merge rules: left<TAB>right<TAB>frequency\n";

    for (const auto& rule : merges_) {
        out << rule.left << '\t'
            << rule.right << '\t'
            << rule.frequency << '\n';
    }
}

void BPETokenizer::load_merges(const std::string& path) {
    std::ifstream in(path);

    if (!in) {
        throw std::runtime_error("Could not open merge file for reading: " + path);
    }

    merges_.clear();
    merge_rank_.clear();
    learned_vocabulary_.clear();

    std::string line;

    while (std::getline(in, line)) {
        if (line.empty() || line.front() == '#') {
            continue;
        }

        const auto first_tab = line.find('\t');

        if (first_tab == std::string::npos) {
            continue;
        }

        const auto second_tab = line.find('\t', first_tab + 1);

        std::string left = line.substr(0, first_tab);
        std::string right;
        std::size_t frequency = 0;

        if (second_tab == std::string::npos) {
            right = line.substr(first_tab + 1);
        } else {
            right = line.substr(first_tab + 1, second_tab - first_tab - 1);

            const auto frequency_text = line.substr(second_tab + 1);
            if (!frequency_text.empty()) {
                try {
                    frequency = static_cast<std::size_t>(std::stoull(frequency_text));
                } catch (...) {
                    frequency = 0;
                }
            }
        }

        MergeRule rule;
        rule.left = std::move(left);
        rule.right = std::move(right);
        rule.merged = rule.left + rule.right;
        rule.rank = merges_.size();
        rule.frequency = frequency;

        merges_.push_back(std::move(rule));
    }

    rebuild_merge_index();

    auto insert_normalized = [this](std::string symbol) {
        if (std::string_view{symbol} == EndOfWord) {
            return;
        }

        if (ends_with(symbol, EndOfWord)) {
            symbol.erase(symbol.size() - EndOfWord.size());
        }

        if (!symbol.empty()) {
            learned_vocabulary_.insert(std::move(symbol));
        }
    };

    for (const auto& rule : merges_) {
        insert_normalized(rule.left);
        insert_normalized(rule.right);
        insert_normalized(rule.merged);
    }
}

const std::vector<BPETokenizer::MergeRule>& BPETokenizer::merges() const noexcept {
    return merges_;
}

std::size_t BPETokenizer::vocabulary_size() const noexcept {
    return learned_vocabulary_.size();
}

std::size_t BPETokenizer::memory_estimate_bytes() const noexcept {
    std::size_t bytes = sizeof(*this);

    bytes += merges_.capacity() * sizeof(MergeRule);

    for (const auto& rule : merges_) {
        bytes += rule.left.capacity();
        bytes += rule.right.capacity();
        bytes += rule.merged.capacity();
    }

    bytes += merge_rank_.bucket_count() * sizeof(void*);

    for (const auto& [key, value] : merge_rank_) {
        bytes += sizeof(key) + key.capacity();
        bytes += sizeof(value);
    }

    bytes += learned_vocabulary_.bucket_count() * sizeof(void*);

    for (const auto& token : learned_vocabulary_) {
        bytes += sizeof(token) + token.capacity();
    }

    return bytes;
}

std::size_t BPETokenizer::max_merges() const noexcept {
    return max_merges_;
}

void BPETokenizer::set_max_merges(std::size_t value) noexcept {
    max_merges_ = value;
}

std::size_t BPETokenizer::min_pair_frequency() const noexcept {
    return min_pair_frequency_;
}

void BPETokenizer::set_min_pair_frequency(std::size_t value) noexcept {
    min_pair_frequency_ = value;
}

std::vector<std::string> BPETokenizer::split_utf8_symbols(std::string_view text) {
    std::vector<std::string> symbols;

    for (std::size_t i = 0; i < text.size();) {
        const auto lead = static_cast<unsigned char>(text[i]);
        std::size_t length = utf8_char_length(lead);

        if (i + length > text.size()) {
            length = 1;
        }

        symbols.emplace_back(text.substr(i, length));
        i += length;
    }

    return symbols;
}

std::string BPETokenizer::pair_key(std::string_view left, std::string_view right) {
    std::string key;
    key.reserve(left.size() + right.size() + 1);
    key.append(left);
    key.push_back('\x1F');
    key.append(right);
    return key;
}

bool BPETokenizer::ends_with(std::string_view value, std::string_view suffix) noexcept {
    if (suffix.size() > value.size()) {
        return false;
    }

    return value.substr(value.size() - suffix.size()) == suffix;
}

void BPETokenizer::merge_pair_in_symbols(std::vector<std::string>& symbols,
                                         std::string_view left,
                                         std::string_view right,
                                         std::string_view merged) {
    if (symbols.size() < 2) {
        return;
    }

    std::vector<std::string> output;
    output.reserve(symbols.size());

    for (std::size_t i = 0; i < symbols.size();) {
        if (i + 1 < symbols.size() &&
            std::string_view{symbols[i]} == left &&
            std::string_view{symbols[i + 1]} == right) {
            output.emplace_back(merged);
            i += 2;
        } else {
            output.push_back(std::move(symbols[i]));
            ++i;
        }
    }

    symbols = std::move(output);
}

std::vector<std::string>
BPETokenizer::strip_end_of_word(const std::vector<std::string>& symbols) {
    std::vector<std::string> output;
    output.reserve(symbols.size());

    for (const auto& symbol : symbols) {
        if (std::string_view{symbol} == EndOfWord) {
            continue;
        }

        std::string token = symbol;

        if (ends_with(token, EndOfWord)) {
            token.erase(token.size() - EndOfWord.size());
        }

        if (!token.empty()) {
            output.push_back(std::move(token));
        }
    }

    return output;
}

void BPETokenizer::rebuild_merge_index() {
    merge_rank_.clear();

    for (const auto& rule : merges_) {
        merge_rank_[pair_key(rule.left, rule.right)] = rule.rank;
    }
}

void BPETokenizer::rebuild_vocabulary(const std::vector<WordEntry>& entries) {
    learned_vocabulary_.clear();

    for (const auto& entry : entries) {
        auto tokens = strip_end_of_word(entry.symbols);

        for (auto& token : tokens) {
            learned_vocabulary_.insert(std::move(token));
        }
    }
}
