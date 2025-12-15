#include "ngram_counter.h"
#include <omp.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <functional>
#include "data_loader.h"
#include <fstream>
#include <filesystem>

namespace fs = std::filesystem;

//Sequenziale
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

//Parallele - TLS based
Histogram count_par_chunk_based_tls(const std::string& directory_path, int n_gram_size, int num_threads)
{
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

    #pragma omp parallel num_threads(num_threads) default(none) \
        shared(doc_count, file_paths, tokenized_docs)
    {
        #pragma omp for schedule(dynamic)
        for (size_t i = 0; i < doc_count; ++i) {

            std::ifstream file(file_paths[i]);
            if (!file.is_open()) continue;

            std::stringstream buffer;
            buffer << file.rdbuf();
            std::string content = buffer.str();
            file.close();

            tokenized_docs[i] = tokenize_text(content);
        }
    }

    size_t total_words_count = 0;
    for (const auto& doc_words : tokenized_docs) {
        total_words_count += doc_words.size();
    }

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

    std::vector<Histogram> local_hists(num_threads);
    const int CHUNK_SIZE = 10000;
    const size_t limit = final_words.size() - n_gram_size;

    #pragma omp parallel num_threads(num_threads) default(none) \
        shared(final_words, n_gram_size, local_hists, limit, CHUNK_SIZE)
    {
        const int tid = omp_get_thread_num();
        Histogram& my_hist = local_hists[tid];

        #pragma omp for nowait schedule(static, CHUNK_SIZE)
        for (size_t i = 0; i <= limit; ++i) {

            std::string n_gram = final_words.at(i);

            for (int j = 1; j < n_gram_size; ++j) {
                n_gram += " " + final_words.at(i+j);
            }
            my_hist[n_gram]++;
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

Histogram count_par_document_level_tls(const std::string& directory_path, int ngram_size, int num_threads) {

    std::vector<fs::path> file_paths;
    fs::path dir_path(directory_path);

    if (!fs::exists(dir_path) || !fs::is_directory(dir_path)) {
        return {};
    }

    try {
        for (const auto& entry : fs::directory_iterator(dir_path)) {
            if (entry.is_regular_file() && entry.path().extension() == ".txt") {
                file_paths.push_back(entry.path());
            }
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Errore filesystem: " << e.what() << std::endl;
        return {};
    }

    if (file_paths.empty()) return {};

    const size_t doc_count = file_paths.size();
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

Histogram count_par_dynamic_locking(const std::string& directory_path, int n_gram_size, int num_threads)
{
    // ----------------------------------------------------
    // FASE 1: RACCOLTA SEQUENZIALE DEI PERCORSI (Setup)
    // ----------------------------------------------------
    std::vector<fs::path> file_paths;
    fs::path dir_path(directory_path);

    if (!fs::exists(dir_path) || !fs::is_directory(dir_path)) {
        return {};
    }

    try {
        for (const auto& entry : fs::directory_iterator(dir_path)) {
            if (entry.is_regular_file() && entry.path().extension() == ".txt") {
                file_paths.push_back(entry.path());
            }
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Errore filesystem: " << e.what() << std::endl;
        return {};
    }

    if (file_paths.empty()) return {};

    const size_t doc_count = file_paths.size();

    // ----------------------------------------------------
    // FASE 2: CONFIGURAZIONE LOCKING A GRANULARITÀ FINE
    // ----------------------------------------------------
    constexpr int NUM_SHARDS = 1024;
    std::vector<Histogram> shards(NUM_SHARDS);
    std::vector<omp_lock_t> shard_locks(NUM_SHARDS);

    for (auto& l : shard_locks)
        omp_init_lock(&l);

    // ----------------------------------------------------
    // FASE 3: I/O + ELABORAZIONE PARALLELA (Locking)
    // ----------------------------------------------------
    #pragma omp parallel num_threads(num_threads) default(none) \
        shared(doc_count, n_gram_size, shards, shard_locks, file_paths)
    {
        // Distribuzione del lavoro sui percorsi dei file (Dynamic per bilanciare I/O variabile)
        #pragma omp for schedule(dynamic)
        for (size_t i = 0; i < doc_count; ++i) {

            // --- FASE 3a: I/O (Lettura Concorrente) ---
            const fs::path& current_path = file_paths[i];

            std::ifstream file(current_path);
            if (!file.is_open()) continue;

            std::stringstream buffer;
            buffer << file.rdbuf();
            std::string content = buffer.str();
            file.close();

            // --- FASE 3b: ELABORAZIONE (Conteggio con Locking) ---

            // Tokenizzazione del contenuto letto
            const std::vector<std::string> words = tokenize_text(content);

            if (words.size() < static_cast<size_t>(n_gram_size)) continue;

            const size_t limit = words.size() - n_gram_size;

            // Loop sequenziale all'interno del documento, sincronizzato con lock
            for (size_t k = 0; k <= limit; ++k) {

                std::string n_gram = words.at(k);

                for (int j = 1; j < n_gram_size; ++j) {
                    n_gram += " " + words.at(k+j);
                }

                // Sincronizzazione con il lock sullo shard corretto
                const size_t h = std::hash<std::string>{}(n_gram);
                const size_t shard_id = h % NUM_SHARDS;

                omp_set_lock(&shard_locks[shard_id]);
                shards[shard_id][n_gram]++;
                omp_unset_lock(&shard_locks[shard_id]);
            }
        }
    }

    // ----------------------------------------------------
    // FASE 4: MERGE SEQUENZIALE E CLEANUP
    // ----------------------------------------------------
    Histogram final_hist;
    // ... (Logica di pre-allocazione omessa per brevità) ...
    for (auto& shard : shards) {
        for (auto& [k, v] : shard)
            final_hist[k] += v;
    }

    for (auto& l : shard_locks)
        omp_destroy_lock(&l);

    return final_hist;
}