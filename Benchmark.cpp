#include "BPETokenizer.hpp"
#include "FSTBPETokenizer.hpp"
#include "MorphologicalFST.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

namespace {
struct BenchmarkResult {
    std::string tokenizer_name;
    std::size_t total_tokens{};
    std::size_t vocabulary_size{};
    double average_tokens_per_word{};
    double runtime_ms{};
    std::size_t memory_estimate_bytes{};
};

std::vector<std::string> make_sample_corpus() {
    return {
        "wasikunapi",
        "wasikunamanta",
        "runakunapi",
        "runakunamanta",
        "rimashani",
        "rimashankichu",

        "kitaplarimizdan",
        "kitaplarimizda",
        "evlerimizden",
        "evlerinizde",
        "uylaringizdan",
        "okullarimizdan",
        "arabalarinizdan",
        "cocuklarimizdan"
    };
}

std::vector<std::string> repeat_corpus(const std::vector<std::string>& corpus,
                                       std::size_t repetitions) {
    std::vector<std::string> output;
    output.reserve(corpus.size() * repetitions);

    for (std::size_t i = 0; i < repetitions; ++i) {
        output.insert(output.end(), corpus.begin(), corpus.end());
    }

    return output;
}

MorphologicalFST make_sample_fst() {
    MorphologicalFST fst;
    fst.set_min_root_symbols(2);

    fst.add_roots({
        "wasi",
        "runa",
        "rima",

        "kitap",
        "ev",
        "uy",
        "okul",
        "araba",
        "cocuk"
    });

    fst.add_suffix_rule("quechua.plural", "kuna", "kuna", 10);
    fst.add_suffix_rule("quechua.locative", "pi", "pi", 10);
    fst.add_suffix_rule("quechua.ablative", "manta", "manta", 10);
    fst.add_suffix_rule("quechua.progressive", "sha", "sha", 10);
    fst.add_suffix_rule("quechua.first_person", "ni", "ni", 10);
    fst.add_suffix_rule("quechua.second_person", "nki", "nki", 10);
    fst.add_suffix_rule("quechua.question", "chu", "chu", 10);

    fst.add_suffix_rule("turkish.plural.back", "lar", "lar", 10);
    fst.add_suffix_rule("turkish.plural.front", "ler", "ler", 10);
    fst.add_suffix_rule("turkish.possessive.1pl", "imiz", "imiz", 10);
    fst.add_suffix_rule("turkish.possessive.2pl", "iniz", "iniz", 10);
    fst.add_suffix_rule("turkish.possessive.2pl.alt", "ingiz", "ingiz", 10);
    fst.add_suffix_rule("turkish.ablative.back", "dan", "dan", 10);
    fst.add_suffix_rule("turkish.ablative.front", "den", "den", 10);
    fst.add_suffix_rule("turkish.locative.back", "da", "da", 10);
    fst.add_suffix_rule("turkish.locative.front", "de", "de", 10);

    return fst;
}

std::string format_tokens(const std::vector<std::string>& tokens) {
    if (tokens.empty()) {
        return "[]";
    }

    std::string output;

    for (const auto& token : tokens) {
        output.push_back('[');
        output += token;
        output += "] ";
    }

    output.pop_back();
    return output;
}

template <typename Tokenizer>
BenchmarkResult evaluate(const std::string& name,
                         const Tokenizer& tokenizer,
                         const std::vector<std::string>& corpus) {
    std::unordered_set<std::string> vocabulary;
    std::size_t total_tokens = 0;

    const auto start = std::chrono::high_resolution_clock::now();

    for (const auto& word : corpus) {
        const auto tokens = tokenizer.tokenize_word(word);
        total_tokens += tokens.size();

        for (const auto& token : tokens) {
            vocabulary.insert(token);
        }
    }

    const auto end = std::chrono::high_resolution_clock::now();

    const double runtime_ms =
        std::chrono::duration<double, std::milli>(end - start).count();

    const double average_tokens =
        corpus.empty()
            ? 0.0
            : static_cast<double>(total_tokens) / static_cast<double>(corpus.size());

    return BenchmarkResult{
        .tokenizer_name = name,
        .total_tokens = total_tokens,
        .vocabulary_size = vocabulary.size(),
        .average_tokens_per_word = average_tokens,
        .runtime_ms = runtime_ms,
        .memory_estimate_bytes = tokenizer.memory_estimate_bytes()
    };
}

void print_result(const BenchmarkResult& result) {
    std::cout << "Tokenizer: " << result.tokenizer_name << '\n';
    std::cout << "Tokens: " << result.total_tokens << '\n';
    std::cout << "Vocabulary: " << result.vocabulary_size << '\n';
    std::cout << "Average Tokens/Word: "
              << std::fixed << std::setprecision(2)
              << result.average_tokens_per_word << '\n';
    std::cout << "Runtime: "
              << std::fixed << std::setprecision(3)
              << result.runtime_ms << " ms\n";
    std::cout << "Memory Estimate: "
              << result.memory_estimate_bytes << " bytes\n";
    std::cout << '\n';
}
}

void run_benchmark_demo() {
    constexpr std::size_t merges = 35;
    constexpr std::size_t min_pair_frequency = 2;
    constexpr std::size_t benchmark_repetitions = 1000;

    const auto training_corpus = make_sample_corpus();
    const auto evaluation_corpus = repeat_corpus(training_corpus, benchmark_repetitions);

    BPETokenizer bpe(merges, min_pair_frequency);
    bpe.train(training_corpus);

    auto fst = make_sample_fst();

    FSTBPETokenizer fst_bpe(fst, merges, min_pair_frequency);
    fst_bpe.train(training_corpus);

    std::cout << "Sample word: wasikunapi\n";
    std::cout << "FST segmentation: "
              << format_tokens(fst.segment_word("wasikunapi")) << '\n';
    std::cout << "BPE tokens: "
              << format_tokens(bpe.tokenize_word("wasikunapi")) << '\n';
    std::cout << "FST-BPE tokens: "
              << format_tokens(fst_bpe.tokenize_word("wasikunapi")) << '\n';
    std::cout << '\n';

    std::cout << "Training words: " << training_corpus.size() << '\n';
    std::cout << "Evaluation words: " << evaluation_corpus.size() << '\n';
    std::cout << "Configured merges: " << merges << '\n';
    std::cout << '\n';

    const auto bpe_result = evaluate("BPE", bpe, evaluation_corpus);
    const auto fst_bpe_result = evaluate("FST-BPE", fst_bpe, evaluation_corpus);

    print_result(bpe_result);
    print_result(fst_bpe_result);
}
