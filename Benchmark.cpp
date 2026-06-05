#include "BPETokenizer.hpp"
#include "FSTBPETokenizer.hpp"
#include "FSTTokenizer.hpp"
#include "MorphologicalFST.hpp"


#include <algorithm>
#include <cctype>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {
struct BenchmarkResult {
    std::string tokenizer_name;
    std::size_t evaluated_words{};
    std::size_t total_tokens{};
    std::size_t observed_vocabulary_size{};
    std::size_t learned_vocabulary_size{};
    std::size_t learned_merges{};
    std::size_t fst_rules{};
    std::size_t fst_roots{};
    double average_tokens_per_word{};

    double training_runtime_ms{};
    double inference_runtime_ms{};
    std::size_t memory_estimate_bytes{};
};

struct CorpusSpec {
    std::string language;
    std::string path;
};

struct CorpusSplit {
    std::vector<std::string> train;
    std::vector<std::string> evaluation;
};

struct ProgramOptions {
    std::vector<CorpusSpec> corpora;
    std::size_t merges = 100;
    std::size_t min_pair_frequency = 2;
    std::size_t max_words = 0;
    std::size_t sample_count = 5;
    double train_ratio = 0.8;
    unsigned int seed = 13;
    bool help = false;
    bool force_demo = false;
};

struct FSTSegmentationStats {
    std::size_t words{};
    std::size_t segmented_words{};
    std::size_t total_segments{};

    [[nodiscard]] double segmented_percent() const {
        if (words == 0) {
            return 0.0;
        }

        return 100.0 * static_cast<double>(segmented_words) / static_cast<double>(words);
    }

    [[nodiscard]] double average_segments_per_word() const {
        if (words == 0) {
            return 0.0;
        }

        return static_cast<double>(total_segments) / static_cast<double>(words);
    }
};

std::string to_lower_ascii(std::string value) {
    for (auto& character : value) {
        const auto byte = static_cast<unsigned char>(character);
        if (byte < 128U) {
            character = static_cast<char>(std::tolower(byte));
        }
    }

    return value;
}

bool is_ascii_word_edge(unsigned char value) {
    return std::isalnum(value) != 0 || value == '\'';
}

bool contains_word_like_character(std::string_view value) {
    for (const auto character : value) {
        const auto byte = static_cast<unsigned char>(character);

        if (byte >= 128U || std::isalpha(byte) != 0) {
            return true;
        }
    }

    return false;
}

std::string normalize_token(std::string token) {
    token = to_lower_ascii(std::move(token));

    std::size_t begin = 0;
    while (begin < token.size()) {
        const auto byte = static_cast<unsigned char>(token[begin]);
        if (byte >= 128U || is_ascii_word_edge(byte)) {
            break;
        }
        ++begin;
    }

    std::size_t end = token.size();
    while (end > begin) {
        const auto byte = static_cast<unsigned char>(token[end - 1]);
        if (byte >= 128U || is_ascii_word_edge(byte)) {
            break;
        }
        --end;
    }

    if (begin >= end) {
        return {};
    }

    std::string normalized = token.substr(begin, end - begin);

    if (!contains_word_like_character(normalized)) {
        return {};
    }

    return normalized;
}

std::vector<std::string> read_words_from_file(const std::string& path,
                                              std::size_t max_words) {
    std::ifstream input(path);

    if (!input) {
        throw std::runtime_error("Could not open corpus file: " + path);
    }

    std::vector<std::string> words;
    std::string raw_token;

    while (input >> raw_token) {
        auto token = normalize_token(std::move(raw_token));

        if (!token.empty()) {
            words.push_back(std::move(token));
        }

        if (max_words > 0 && words.size() >= max_words) {
            break;
        }
    }

    if (words.empty()) {
        throw std::runtime_error("No usable word tokens were found in: " + path);
    }

    return words;
}

CorpusSplit split_corpus(std::vector<std::string> words,
                         double train_ratio,
                         unsigned int seed) {
    if (words.empty()) {
        return {};
    }

    if (words.size() == 1) {
        return CorpusSplit{words, words};
    }

    std::mt19937 generator(seed);
    std::shuffle(words.begin(), words.end(), generator);

    if (train_ratio <= 0.0 || train_ratio >= 1.0) {
        train_ratio = 0.8;
    }

    std::size_t train_size = static_cast<std::size_t>(
        static_cast<double>(words.size()) * train_ratio
    );

    train_size = std::max<std::size_t>(1, train_size);
    train_size = std::min<std::size_t>(train_size, words.size() - 1);

    CorpusSplit split;
    split.train.assign(words.begin(), words.begin() + static_cast<std::ptrdiff_t>(train_size));
    split.evaluation.assign(words.begin() + static_cast<std::ptrdiff_t>(train_size), words.end());

    return split;
}

void add_english_rules(MorphologicalFST& fst) {
    fst.set_min_root_symbols(3);

    fst.add_suffix_rule("english.progressive", "ing", "ing", 10);
    fst.add_suffix_rule("english.past", "ed", "ed", 10);
    fst.add_suffix_rule("english.adverb", "ly", "ly", 8);
    fst.add_suffix_rule("english.plural.s", "s", "s", 4);
    fst.add_suffix_rule("english.plural.es", "es", "es", 8);
    fst.add_suffix_rule("english.plural.ies", "ies", "ies", 9);
    fst.add_suffix_rule("english.comparative", "er", "er", 8);
    fst.add_suffix_rule("english.superlative", "est", "est", 8);
    fst.add_suffix_rule("english.nounness", "ness", "ness", 10);
    fst.add_suffix_rule("english.nounment", "ment", "ment", 10);
    fst.add_suffix_rule("english.tion", "tion", "tion", 10);
    fst.add_suffix_rule("english.sion", "sion", "sion", 10);
    fst.add_suffix_rule("english.able", "able", "able", 10);
    fst.add_suffix_rule("english.ible", "ible", "ible", 10);
    fst.add_suffix_rule("english.ality", "ality", "ality", 10);
}

void add_quechua_rules(MorphologicalFST& fst) {
    fst.set_min_root_symbols(2);

    fst.add_suffix_rule("quechua.plural", "kuna", "kuna", 10);
    fst.add_suffix_rule("quechua.locative", "pi", "pi", 10);
    fst.add_suffix_rule("quechua.ablative", "manta", "manta", 10);
    fst.add_suffix_rule("quechua.accusative", "ta", "ta", 7);
    fst.add_suffix_rule("quechua.genitive", "pa", "pa", 7);
    fst.add_suffix_rule("quechua.instrumental", "wan", "wan", 8);
    fst.add_suffix_rule("quechua.limitative", "kama", "kama", 8);
    fst.add_suffix_rule("quechua.similative", "hina", "hina", 8);
    fst.add_suffix_rule("quechua.restrictive", "lla", "lla", 7);
    fst.add_suffix_rule("quechua.topic", "qa", "qa", 6);
    fst.add_suffix_rule("quechua.progressive", "sha", "sha", 10);
    fst.add_suffix_rule("quechua.first_person", "ni", "ni", 10);
    fst.add_suffix_rule("quechua.second_person", "nki", "nki", 10);
    fst.add_suffix_rule("quechua.first_person_plural", "nchik", "nchik", 10);
    fst.add_suffix_rule("quechua.second_possessive", "yki", "yki", 10);
    fst.add_suffix_rule("quechua.first_possessive", "y", "y", 4);
    fst.add_suffix_rule("quechua.question", "chu", "chu", 10);
}

void add_turkish_rules(MorphologicalFST& fst) {
    fst.set_min_root_symbols(2);

    fst.add_suffix_rule("turkish.plural.back", "lar", "lar", 10);
    fst.add_suffix_rule("turkish.plural.front", "ler", "ler", 10);

    fst.add_suffix_rule("turkish.possessive.1pl.ascii", "imiz", "imiz", 10);
    fst.add_suffix_rule("turkish.possessive.1pl.back_unrounded", "ımız", "ımız", 10);
    fst.add_suffix_rule("turkish.possessive.1pl.back_rounded", "umuz", "umuz", 10);
    fst.add_suffix_rule("turkish.possessive.1pl.front_rounded", "ümüz", "ümüz", 10);

    fst.add_suffix_rule("turkish.possessive.2pl.ascii", "iniz", "iniz", 10);
    fst.add_suffix_rule("turkish.possessive.2pl.back_unrounded", "ınız", "ınız", 10);
    fst.add_suffix_rule("turkish.possessive.2pl.back_rounded", "unuz", "unuz", 10);
    fst.add_suffix_rule("turkish.possessive.2pl.front_rounded", "ünüz", "ünüz", 10);
    fst.add_suffix_rule("turkish.possessive.2pl.alt", "ingiz", "ingiz", 10);
    fst.add_suffix_rule("turkish.possessive.2pl.alt_back", "ınız", "ınız", 10);

    fst.add_suffix_rule("turkish.ablative.back", "dan", "dan", 10);
    fst.add_suffix_rule("turkish.ablative.front", "den", "den", 10);
    fst.add_suffix_rule("turkish.ablative.back_devoiced", "tan", "tan", 10);
    fst.add_suffix_rule("turkish.ablative.front_devoiced", "ten", "ten", 10);

    fst.add_suffix_rule("turkish.locative.back", "da", "da", 8);
    fst.add_suffix_rule("turkish.locative.front", "de", "de", 8);
    fst.add_suffix_rule("turkish.locative.back_devoiced", "ta", "ta", 8);
    fst.add_suffix_rule("turkish.locative.front_devoiced", "te", "te", 8);

    fst.add_suffix_rule("turkish.genitive.ascii", "nin", "nin", 8);
    fst.add_suffix_rule("turkish.genitive.back_unrounded", "nın", "nın", 8);
    fst.add_suffix_rule("turkish.genitive.back_rounded", "nun", "nun", 8);
    fst.add_suffix_rule("turkish.genitive.front_rounded", "nün", "nün", 8);

    fst.add_suffix_rule("turkish.dative.back", "ya", "ya", 7);
    fst.add_suffix_rule("turkish.dative.front", "ye", "ye", 7);
    fst.add_suffix_rule("turkish.dative.back_plain", "a", "a", 4);
    fst.add_suffix_rule("turkish.dative.front_plain", "e", "e", 4);

    fst.add_suffix_rule("turkish.progressive.ascii", "iyor", "iyor", 10);
    fst.add_suffix_rule("turkish.progressive.back_unrounded", "ıyor", "ıyor", 10);
    fst.add_suffix_rule("turkish.progressive.back_rounded", "uyor", "uyor", 10);
    fst.add_suffix_rule("turkish.progressive.front_rounded", "üyor", "üyor", 10);
    fst.add_suffix_rule("turkish.future.back", "acak", "acak", 10);
    fst.add_suffix_rule("turkish.future.front", "ecek", "ecek", 10);
    fst.add_suffix_rule("turkish.evidential.ascii", "mis", "mis", 8);
    fst.add_suffix_rule("turkish.evidential.front", "miş", "miş", 8);
    fst.add_suffix_rule("turkish.evidential.back", "mış", "mış", 8);
    fst.add_suffix_rule("turkish.evidential.back_rounded", "muş", "muş", 8);
    fst.add_suffix_rule("turkish.evidential.front_rounded", "müş", "müş", 8);
}

MorphologicalFST make_fst_for_language(std::string_view language_name) {
    const auto language = to_lower_ascii(std::string{language_name});
    MorphologicalFST fst;

    if (language == "english" || language == "en" || language == "eng") {
        add_english_rules(fst);
        return fst;
    }

    if (language == "quechua" || language == "qu" || language == "que" ||
        language == "quy" || language == "aymara") {
        add_quechua_rules(fst);
        return fst;
    }

    if (language == "turkish" || language == "tr" || language == "tur" ||
        language == "azerbaijani" || language == "az" || language == "uzbek" ||
        language == "uz") {
        add_turkish_rules(fst);
        return fst;
    }

    if (language == "generic" || language == "agglutinative" || language == "agg") {
        add_quechua_rules(fst);
        add_turkish_rules(fst);
        return fst;
    }

    // Unknown languages still run through FST and FST-BPE, but without rules
    // the FST emits each word unchanged, making them useful control conditions.
    return fst;
}

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

    add_quechua_rules(fst);
    add_turkish_rules(fst);

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

template <typename Function>
double measure_ms(Function&& function) {
    const auto start = std::chrono::high_resolution_clock::now();
    function();
    const auto end = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(end - start).count();
}

template <typename Tokenizer>
BenchmarkResult evaluate(const std::string& name,
                         const Tokenizer& tokenizer,
                         const std::vector<std::string>& corpus,
                         double training_runtime_ms,
                         std::size_t learned_vocabulary_size,
                         std::size_t learned_merges,
                         std::size_t fst_rules = 0,
                         std::size_t fst_roots = 0) {

    std::unordered_set<std::string> observed_vocabulary;
    std::size_t total_tokens = 0;

    const double inference_runtime_ms = measure_ms([&] {
        for (const auto& word : corpus) {
            const auto tokens = tokenizer.tokenize_word(word);
            total_tokens += tokens.size();

            for (const auto& token : tokens) {
                observed_vocabulary.insert(token);
            }
        }
    });

    const double average_tokens =
        corpus.empty()
            ? 0.0
            : static_cast<double>(total_tokens) / static_cast<double>(corpus.size());

    return BenchmarkResult{
        .tokenizer_name = name,
        .evaluated_words = corpus.size(),
        .total_tokens = total_tokens,
        .observed_vocabulary_size = observed_vocabulary.size(),
        .learned_vocabulary_size = learned_vocabulary_size,
        .learned_merges = learned_merges,
        .fst_rules = fst_rules,
        .fst_roots = fst_roots,
        .average_tokens_per_word = average_tokens,

        .training_runtime_ms = training_runtime_ms,
        .inference_runtime_ms = inference_runtime_ms,
        .memory_estimate_bytes = tokenizer.memory_estimate_bytes()
    };
}

FSTSegmentationStats compute_fst_segmentation_stats(const MorphologicalFST& fst,
                                                    const std::vector<std::string>& words) {
    FSTSegmentationStats stats;
    stats.words = words.size();

    for (const auto& word : words) {
        const auto segments = fst.segment_word(word);
        stats.total_segments += segments.size();

        if (segments.size() > 1) {
            ++stats.segmented_words;
        }
    }

    return stats;
}

void print_result(const BenchmarkResult& result) {
    std::cout << "Tokenizer: " << result.tokenizer_name << '\n';
    std::cout << "Evaluation Words: " << result.evaluated_words << '\n';
    std::cout << "Tokens: " << result.total_tokens << '\n';
    std::cout << "Observed Vocabulary: " << result.observed_vocabulary_size << '\n';
    std::cout << "Learned Vocabulary: " << result.learned_vocabulary_size << '\n';
    std::cout << "Learned Merges: " << result.learned_merges << '\n';
    std::cout << "FST Rules: " << result.fst_rules << '\n';
    std::cout << "FST Learned Roots: " << result.fst_roots << '\n';
    std::cout << "Average Tokens/Word: "

              << std::fixed << std::setprecision(2)
              << result.average_tokens_per_word << '\n';
    std::cout << "Training Runtime: "
              << std::fixed << std::setprecision(3)
              << result.training_runtime_ms << " ms\n";
    std::cout << "Inference Runtime: "
              << std::fixed << std::setprecision(3)
              << result.inference_runtime_ms << " ms\n";
    std::cout << "Memory Estimate: "
              << result.memory_estimate_bytes << " bytes\n";
    std::cout << '\n';
}

void print_delta(const BenchmarkResult& baseline,
                 const BenchmarkResult& candidate) {
    const auto token_delta = static_cast<long long>(candidate.total_tokens) -
                             static_cast<long long>(baseline.total_tokens);
    const double avg_delta = candidate.average_tokens_per_word - baseline.average_tokens_per_word;
    const double runtime_ratio =
        baseline.inference_runtime_ms == 0.0
            ? 0.0
            : candidate.inference_runtime_ms / baseline.inference_runtime_ms;

    std::cout << "Difference: " << candidate.tokenizer_name
              << " minus " << baseline.tokenizer_name << '\n';
    std::cout << "Token Delta: " << token_delta << '\n';
    std::cout << "Average Tokens/Word Delta: "
              << std::fixed << std::setprecision(2) << avg_delta << '\n';
    std::cout << "Inference Runtime Ratio: "
              << std::fixed << std::setprecision(2) << runtime_ratio << "x\n";
    std::cout << '\n';
}


void print_samples(const std::vector<std::string>& words,
                   const BPETokenizer& bpe,
                   const FSTTokenizer& fst,
                   const FSTBPETokenizer& fst_bpe,
                   std::size_t sample_count) {

    if (sample_count == 0 || words.empty()) {
        return;
    }

    std::cout << "Sample tokenizations:\n";

    const std::size_t count = std::min(sample_count, words.size());
    for (std::size_t i = 0; i < count; ++i) {
        const auto& word = words[i];

        std::cout << "Word: " << word << '\n';
        std::cout << "  BPE: "
                  << format_tokens(bpe.tokenize_word(word)) << '\n';
        std::cout << "  FST: "
                  << format_tokens(fst.tokenize_word(word)) << '\n';
        std::cout << "  FST-BPE: "
                  << format_tokens(fst_bpe.tokenize_word(word)) << '\n';

    }

    std::cout << '\n';
}

std::size_t parse_size(std::string_view text, std::string_view option_name) {
    try {
        return static_cast<std::size_t>(std::stoull(std::string{text}));
    } catch (...) {
        throw std::invalid_argument("Invalid integer for " + std::string{option_name} + ": " + std::string{text});
    }
}

double parse_double(std::string_view text, std::string_view option_name) {
    try {
        return std::stod(std::string{text});
    } catch (...) {
        throw std::invalid_argument("Invalid number for " + std::string{option_name} + ": " + std::string{text});
    }
}

std::string require_next(int& index,
                         int argc,
                         char** argv,
                         std::string_view option_name) {
    if (index + 1 >= argc) {
        throw std::invalid_argument("Missing value for " + std::string{option_name});
    }

    ++index;
    return argv[index];
}

CorpusSpec parse_corpus_spec(const std::string& value) {
    const auto separator = value.find(':');

    if (separator == std::string::npos || separator == 0 || separator + 1 >= value.size()) {
        throw std::invalid_argument(
            "Corpus specs must have the form language:path, for example turkish:data/tr.txt"
        );
    }

    return CorpusSpec{
        .language = value.substr(0, separator),
        .path = value.substr(separator + 1)
    };
}

ProgramOptions parse_options(int argc, char** argv) {
    ProgramOptions options;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            options.help = true;
        } else if (arg == "--demo") {
            options.force_demo = true;
        } else if (arg == "--merges") {
            options.merges = parse_size(require_next(i, argc, argv, arg), arg);
        } else if (arg == "--min-frequency" || arg == "--min-pair-frequency") {
            options.min_pair_frequency = parse_size(require_next(i, argc, argv, arg), arg);
        } else if (arg == "--max-words") {
            options.max_words = parse_size(require_next(i, argc, argv, arg), arg);
        } else if (arg == "--samples") {
            options.sample_count = parse_size(require_next(i, argc, argv, arg), arg);
        } else if (arg == "--train-ratio") {
            options.train_ratio = parse_double(require_next(i, argc, argv, arg), arg);
        } else if (arg == "--seed") {
            options.seed = static_cast<unsigned int>(parse_size(require_next(i, argc, argv, arg), arg));
        } else if (arg == "--english" || arg == "--en") {
            options.corpora.push_back(CorpusSpec{"english", require_next(i, argc, argv, arg)});
        } else if (arg == "--turkish" || arg == "--tr") {
            options.corpora.push_back(CorpusSpec{"turkish", require_next(i, argc, argv, arg)});
        } else if (arg == "--quechua" || arg == "--qu") {
            options.corpora.push_back(CorpusSpec{"quechua", require_next(i, argc, argv, arg)});
        } else if (arg == "--corpus") {
            options.corpora.push_back(parse_corpus_spec(require_next(i, argc, argv, arg)));
        } else if (arg.find(':') != std::string::npos) {
            options.corpora.push_back(parse_corpus_spec(arg));
        } else {
            throw std::invalid_argument("Unknown argument: " + arg);
        }
    }

    return options;
}

void print_usage() {
    std::cout << "Usage:\n";
    std::cout << "  ./tokenizer_demo --english data/en.txt --turkish data/tr.txt --quechua data/qu.txt [options]\n";
    std::cout << "  ./tokenizer_demo --corpus english:data/en.txt --corpus turkish:data/tr.txt [options]\n";
    std::cout << "  ./tokenizer_demo english:data/en.txt turkish:data/tr.txt quechua:data/qu.txt [options]\n\n";

    std::cout << "Input files may contain one word per line or normal whitespace-separated text.\n\n";

    std::cout << "Options:\n";
    std::cout << "  --merges N              Number of BPE merge iterations. Default: 100\n";
    std::cout << "  --min-frequency N       Minimum adjacent-pair frequency. Default: 2\n";
    std::cout << "  --train-ratio R         Train/evaluation split ratio. Default: 0.8\n";
    std::cout << "  --max-words N           Read at most N words per file. Default: all\n";
    std::cout << "  --samples N             Print N sample tokenizations per language. Default: 5\n";
    std::cout << "  --seed N                Deterministic shuffle seed. Default: 13\n";
    std::cout << "  --demo                  Run the built-in toy demo\n";
    std::cout << "  --help                  Show this message\n\n";

    std::cout << "Supported FST rule presets: english/en, turkish/tr, quechua/qu, generic/agglutinative.\n";
    std::cout << "Unknown language labels are allowed, but FST and FST-BPE will use an empty FST and act as controls.\n";
}

void run_language_benchmark(const CorpusSpec& spec,
                            const ProgramOptions& options) {
    std::cout << "============================================================\n";
    std::cout << "Language: " << spec.language << '\n';
    std::cout << "File: " << spec.path << '\n';

    auto words = read_words_from_file(spec.path, options.max_words);
    const auto total_loaded_words = words.size();
    auto split = split_corpus(std::move(words), options.train_ratio, options.seed);

    std::cout << "Loaded Words: " << total_loaded_words << '\n';
    std::cout << "Training Words: " << split.train.size() << '\n';
    std::cout << "Evaluation Words: " << split.evaluation.size() << '\n';
    std::cout << "Configured Merges: " << options.merges << '\n';
    std::cout << "Minimum Pair Frequency: " << options.min_pair_frequency << '\n';

    auto fst_rules = make_fst_for_language(spec.language);
    FSTTokenizer fst_tokenizer(fst_rules);
    const double fst_training_ms = measure_ms([&] {
        fst_tokenizer.train(split.train);
    });

    const auto fst_train_stats = compute_fst_segmentation_stats(fst_tokenizer.fst(), split.train);
    const auto fst_eval_stats = compute_fst_segmentation_stats(fst_tokenizer.fst(), split.evaluation);

    std::cout << "FST Rules: " << fst_tokenizer.fst().rules().size() << '\n';
    std::cout << "FST Learned Roots: " << fst_tokenizer.fst().root_count() << '\n';
    std::cout << "FST Segmented Training Words: "
              << fst_train_stats.segmented_words << '/' << fst_train_stats.words
              << " (" << std::fixed << std::setprecision(2)
              << fst_train_stats.segmented_percent() << "%)\n";
    std::cout << "FST Segmented Evaluation Words: "
              << fst_eval_stats.segmented_words << '/' << fst_eval_stats.words
              << " (" << std::fixed << std::setprecision(2)
              << fst_eval_stats.segmented_percent() << "%)\n";
    std::cout << "Average FST Segments/Evaluation Word: "
              << std::fixed << std::setprecision(2)
              << fst_eval_stats.average_segments_per_word() << "\n\n";

    BPETokenizer bpe(options.merges, options.min_pair_frequency);
    const double bpe_training_ms = measure_ms([&] {
        bpe.train(split.train);
    });

    FSTBPETokenizer fst_bpe(fst_tokenizer.fst(), options.merges, options.min_pair_frequency);
    const double fst_bpe_bpe_training_ms = measure_ms([&] {
        fst_bpe.train(split.train);
    });
    const double fst_bpe_training_ms = fst_training_ms + fst_bpe_bpe_training_ms;


    const auto bpe_result = evaluate("BPE",
                                     bpe,
                                     split.evaluation,
                                     bpe_training_ms,
                                     bpe.vocabulary_size(),
                                     bpe.merges().size());

    const auto fst_result = evaluate("FST",
                                     fst_tokenizer,
                                     split.evaluation,
                                     fst_training_ms,
                                     fst_tokenizer.vocabulary_size(),
                                     0,
                                     fst_tokenizer.fst().rules().size(),
                                     fst_tokenizer.fst().root_count());

    const auto fst_bpe_result = evaluate("FST-BPE",
                                         fst_bpe,
                                         split.evaluation,
                                         fst_bpe_training_ms,
                                         fst_bpe.vocabulary_size(),
                                         fst_bpe.bpe().merges().size(),
                                         fst_bpe.fst().rules().size(),
                                         fst_bpe.fst().root_count());

    print_result(bpe_result);
    print_result(fst_result);
    print_result(fst_bpe_result);
    print_delta(bpe_result, fst_result);
    print_delta(bpe_result, fst_bpe_result);
    print_delta(fst_result, fst_bpe_result);
    print_samples(split.evaluation, bpe, fst_tokenizer, fst_bpe, options.sample_count);

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

    FSTTokenizer fst_tokenizer(fst);
    fst_tokenizer.train(training_corpus);

    FSTBPETokenizer fst_bpe(fst_tokenizer.fst(), merges, min_pair_frequency);
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
    std::cout << "FST rules: " << fst_tokenizer.fst().rules().size() << '\n';
    std::cout << "FST learned roots: " << fst_tokenizer.fst().root_count() << '\n';
    std::cout << '\n';

    const auto bpe_result = evaluate("BPE",
                                     bpe,
                                     evaluation_corpus,
                                     0.0,
                                     bpe.vocabulary_size(),
                                     bpe.merges().size());

    const auto fst_result = evaluate("FST",
                                     fst_tokenizer,
                                     evaluation_corpus,
                                     0.0,
                                     fst_tokenizer.vocabulary_size(),
                                     0,
                                     fst_tokenizer.fst().rules().size(),
                                     fst_tokenizer.fst().root_count());

    const auto fst_bpe_result = evaluate("FST-BPE",
                                         fst_bpe,
                                         evaluation_corpus,
                                         0.0,
                                         fst_bpe.vocabulary_size(),
                                         fst_bpe.bpe().merges().size(),
                                         fst_bpe.fst().rules().size(),
                                         fst_bpe.fst().root_count());

    print_result(bpe_result);
    print_result(fst_result);
    print_result(fst_bpe_result);
    print_delta(bpe_result, fst_result);
    print_delta(bpe_result, fst_bpe_result);
    print_delta(fst_result, fst_bpe_result);
}

void run_benchmark_cli(int argc, char** argv) {
    const auto options = parse_options(argc, argv);

    if (options.help) {
        print_usage();
        return;
    }

    if (options.force_demo || options.corpora.empty()) {
        if (options.corpora.empty() && !options.force_demo) {
            std::cout << "No corpus files were provided; running built-in demo. Use --help for file mode.\n\n";
        }

        run_benchmark_demo();
        return;
    }

    std::cout << "Multilingual file benchmark\n";
    std::cout << "Corpora: " << options.corpora.size() << '\n';
    std::cout << "Merges: " << options.merges << '\n';
    std::cout << "Minimum Pair Frequency: " << options.min_pair_frequency << '\n';
    std::cout << "Train Ratio: " << std::fixed << std::setprecision(2) << options.train_ratio << '\n';
    if (options.max_words > 0) {
        std::cout << "Max Words/File: " << options.max_words << '\n';
    }
    std::cout << '\n';

    for (const auto& spec : options.corpora) {
        run_language_benchmark(spec, options);
    }
}
