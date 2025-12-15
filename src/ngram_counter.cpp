#include "ngram_counter.h"
#include <omp.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <functional>
#include "data_loader.h"
#include <fstream>
#include <filesystem>
#include <unordered_map>
#include <string>
#include <sstream>
#include <iterator>

namespace fs = std::filesystem;

// Sequenziale

Histogram count_seq(const std::vector<std::string>& words, const int n_gram_size) {
    Histogram hist;
    if (words.size() < static_cast<size_t>(n_gram_size)) return hist;

    for (size_t i = 0; i <= words.size() - n_gram_size; ++i) {
        std::string n_gram = words[i];
        for (int j = 1; j < n_gram_size; ++j) {
            n_gram += " " + words[i+j];
        }
        hist[n_gram]++;
    }
    return hist;
}

// Parallelo

Histogram count_par_chunk_based_tls(const std::string& directory_path, int n_gram_size, int num_threads, int multiplier)
{
    namespace fs = std::filesystem;
    std::vector<fs::path> file_paths;
    fs::path dir_path(directory_path);

    if (!fs::exists(dir_path) || !fs::is_directory(dir_path))
        return {};

    for (const auto& entry : fs::directory_iterator(dir_path)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            file_paths.push_back(entry.path());
        }
    }
    if (file_paths.empty()) return {};

    const size_t doc_count = file_paths.size();
    std::vector<std::vector<std::string>> tokenized_docs(doc_count);

    #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for (size_t i = 0; i < doc_count; ++i) {
        std::ifstream file(file_paths[i]);
        if (!file.is_open()) continue;

        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string content = buffer.str();
        file.close();

        tokenized_docs[i] = tokenize_text(content);
    }

    size_t total_words_count = 0;
    for (const auto& doc_words : tokenized_docs)
        total_words_count += doc_words.size();

    std::vector<std::string> final_words;
    final_words.reserve(total_words_count);
    for (auto& doc_words : tokenized_docs) {
        final_words.insert(final_words.end(),
                           std::make_move_iterator(doc_words.begin()),
                           std::make_move_iterator(doc_words.end()));
        doc_words.clear();
        doc_words.shrink_to_fit();
    }
    tokenized_docs.clear();

    if (final_words.size() < static_cast<size_t>(n_gram_size)) return {};

    std::vector<std::string> final_words_scaled;
    final_words_scaled.reserve(final_words.size() * multiplier);
    for (int m = 0; m < multiplier; ++m) {
        final_words_scaled.insert(final_words_scaled.end(),
                                  final_words.begin(),
                                  final_words.end());
    }

    std::vector<Histogram> local_hists(num_threads);
    const int CHUNK_SIZE = 10000;
    const size_t limit = final_words_scaled.size() - n_gram_size;

    #pragma omp parallel num_threads(num_threads) default(none) \
        shared(final_words_scaled, n_gram_size, local_hists, limit, CHUNK_SIZE)
    {
        int tid = omp_get_thread_num();
        Histogram& my_hist = local_hists[tid];

        #pragma omp for schedule(static, CHUNK_SIZE) nowait
        for (size_t i = 0; i <= limit; ++i) {
            std::string n_gram = final_words_scaled[i];
            for (int j = 1; j < n_gram_size; ++j)
                n_gram += " " + final_words_scaled[i + j];
            my_hist[n_gram]++;
        }
    }

    Histogram final_hist;
    size_t total_unique_elements = 0;
    for (const auto& hist : local_hists) total_unique_elements += hist.size();
    final_hist.reserve(total_unique_elements);

    for (const auto& hist : local_hists)
        for (const auto& [key, count] : hist)
            final_hist[key] += count;

    return final_hist;
}
Histogram count_par_document_level_tls(const std::string& directory_path, int ngram_size, int num_threads, int multiplier) {

    namespace fs = std::filesystem;
    std::vector<fs::path> file_paths;
    fs::path dir_path(directory_path);

    if (!fs::exists(dir_path) || !fs::is_directory(dir_path)) {
        return {};
    }

    for (const auto& entry : fs::directory_iterator(dir_path)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            file_paths.push_back(entry.path());
        }
    }

    if (file_paths.empty()) return {};

    const size_t doc_count = file_paths.size();

    std::vector<std::vector<std::string>> tokenized_docs(doc_count);
    for (size_t i = 0; i < doc_count; ++i) {
        std::ifstream file(file_paths[i]);
        if (!file.is_open()) continue;
        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string content = buffer.str();
        file.close();

        tokenized_docs[i] = tokenize_text(content);
    }

    std::vector<std::string> all_words;
    size_t total_words = 0;
    for (int m = 0; m < multiplier; ++m) {
        for (const auto& doc_words : tokenized_docs) {
            total_words += doc_words.size();
        }
    }
    all_words.reserve(total_words);

    for (int m = 0; m < multiplier; ++m) {
        for (const auto& doc_words : tokenized_docs) {
            all_words.insert(all_words.end(), doc_words.begin(), doc_words.end());
        }
    }

    if (all_words.size() < static_cast<size_t>(ngram_size)) return {};

    std::vector<Histogram> local_hists(num_threads);

    #pragma omp parallel num_threads(num_threads) default(none) \
        shared(doc_count, ngram_size, local_hists, file_paths)
    {
        int tid = omp_get_thread_num();
        Histogram& my_hist = local_hists[tid];

        #pragma omp for schedule(dynamic)
        for (size_t i = 0; i < doc_count; ++i) {

            const fs::path& current_path = file_paths[i];

            std::ifstream file(current_path);
            if (!file.is_open()) continue;

            std::stringstream buffer;
            buffer << file.rdbuf();
            std::string content = buffer.str();
            file.close();

            const std::vector<std::string> words = tokenize_text(content);

            if (words.size() < static_cast<size_t>(ngram_size)) continue;

            const size_t limit = words.size() - ngram_size;

            for (size_t k = 0; k <= limit; ++k) {

                std::string n_gram = words.at(k);

                for (int j = 1; j < ngram_size; ++j) {
                    n_gram += " " + words.at(k+j);
                }

                my_hist[n_gram]++;
            }
        }
    }

    Histogram final_hist;
    size_t total_unique_elements = 0;

    for (const auto& current_hist : local_hists) {
        total_unique_elements += current_hist.size();
    }
    final_hist.reserve(total_unique_elements);

    for (const auto& current_hist : local_hists) {
        for (const auto& [fst, snd] : current_hist) {
            final_hist[fst] += snd;
        }
    }
    
    return final_hist;
}
Histogram count_par_fine_grained_locking(const std::string& directory_path, int n_gram_size, int num_threads, int multiplier)
{
    namespace fs = std::filesystem;
    std::vector<fs::path> file_paths;
    fs::path dir_path(directory_path);

    if (!fs::exists(dir_path) || !fs::is_directory(dir_path)) {
        return {};
    }

    for (const auto& entry : fs::directory_iterator(dir_path)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            file_paths.push_back(entry.path());
        }
    }

    if (file_paths.empty()) return {};

    std::vector<std::vector<std::string>> tokenized_docs;
    for (const auto& path : file_paths) {
        std::ifstream file(path);
        if (!file.is_open()) continue;
        std::stringstream buffer;
        buffer << file.rdbuf();
        std::string content = buffer.str();
        file.close();
        tokenized_docs.push_back(tokenize_text(content));
    }

    std::vector<std::string> all_words;
    size_t total_words = 0;
    for (int m = 0; m < multiplier; ++m)
        for (const auto& doc_words : tokenized_docs)
            total_words += doc_words.size();
    all_words.reserve(total_words);

    for (int m = 0; m < multiplier; ++m)
        for (const auto& doc_words : tokenized_docs)
            all_words.insert(all_words.end(), doc_words.begin(), doc_words.end());

    if (all_words.size() < static_cast<size_t>(n_gram_size)) return {};

    constexpr int NUM_SHARDS = 1024;
    std::vector<Histogram> shards(NUM_SHARDS);
    std::vector<omp_lock_t> shard_locks(NUM_SHARDS);
    for (auto& l : shard_locks)
        omp_init_lock(&l);

    const size_t limit = all_words.size() - n_gram_size;

    #pragma omp parallel num_threads(num_threads) default(none) \
        shared(all_words, n_gram_size, shards, shard_locks, limit)
    {
        #pragma omp for schedule(static)
        for (size_t i = 0; i <= limit; ++i) {

            std::string n_gram = all_words[i];
            for (int j = 1; j < n_gram_size; ++j)
                n_gram += " " + all_words[i + j];

            const size_t h = std::hash<std::string>{}(n_gram);
            const size_t shard_id = h % NUM_SHARDS;

            omp_set_lock(&shard_locks[shard_id]);
            shards[shard_id][n_gram]++;
            omp_unset_lock(&shard_locks[shard_id]);
        }
    }

    Histogram final_hist;
    size_t total_unique_elements = 0;
    for (const auto& shard : shards)
        total_unique_elements += shard.size();
    final_hist.reserve(total_unique_elements);

    for (const auto& shard : shards)
        for (const auto& [k, v] : shard)
            final_hist[k] += v;

    for (auto& l : shard_locks)
        omp_destroy_lock(&l);

    return final_hist;
}
