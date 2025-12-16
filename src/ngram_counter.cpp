#include "ngram_counter.h"
#include <omp.h>
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

void count_par_singleReader_Worker_TLS(Histogram& hist, int n_gram_size, int max_iter)
{
     std::vector<std::string> texts;

#pragma omp parallel shared(texts, max_iter, n_gram_size, hist) default(none)
    {
         Histogram thread_word_hist;

        // Iterate over the texts
        for (int k=0;k<max_iter;k++) {
            for (const auto &document: std::filesystem::directory_iterator("data/Texts")) {

                // Obtain the path of the next file
                std::string document_path = document.path().string();
                std::ifstream file(document_path);

                #pragma omp single
                {   // Open the file and read the document
                    if (file.is_open()) {
                        std::string content;

                        std::string line;
                        std::stringstream buffer;

                        buffer << file.rdbuf();
                        content = buffer.str();
                        texts.push_back(content);

                        file.close();
                    }
                    else {
                        printf("Impossible open the file: %s", document_path.c_str());
                    }
                }
            }
        }

#pragma omp for nowait schedule(dynamic,1)
        for (auto & text : texts) {
            UpdateHistogramWord(thread_word_hist, text, n_gram_size);
        }


#pragma omp critical (word)
        {
            for (const auto & [fst, snd]: thread_word_hist) {
                hist[fst] += snd;
            }
        }

   }
}

void count_par_onTheFly_parallelIO(Histogram& hist,const int n_gram_size, int max_iter){

    // Count the number of texts in the folder
    int doc_count = 0;
    for ([[maybe_unused]] const auto &document: std::filesystem::directory_iterator("data/Texts")) {
        doc_count++;
    }

    # pragma omp parallel shared(doc_count, max_iter, n_gram_size, hist) default(none)
    {
        Histogram thread_word_histogram;

        for (int k=0; k < max_iter; k++) {
            # pragma omp for nowait schedule(dynamic,1)
            for (int i = 0; i < doc_count; i++) {

                std::string document_path = "data/Texts/" + std::to_string(i) + ".txt";

                if (std::ifstream file(document_path); file.is_open()) {

                    std::stringstream buffer;
                    buffer << file.rdbuf();
                    std::string content = buffer.str();

                    UpdateHistogramWord(thread_word_histogram, content, n_gram_size);

                    file.close();
                } else {
                    printf("Impossible open the file: %s", document_path.c_str());
                }
            }
        }

        #pragma omp critical (word)
        {
            for (const auto & [fst, snd]: thread_word_histogram) {
                hist[fst] += snd;
            }
        }

    }
}

void count_par_hybrid_preload_TLS(Histogram& hist, int n_gram_size, int max_iter){

    int doc_count = 0;
    for ([[maybe_unused]] const auto &document: std::filesystem::directory_iterator("data/Texts")) {
        doc_count++;
    }

#pragma omp parallel shared(doc_count, max_iter, n_gram_size, hist) default(none)
    {
        std::vector<std::string> texts;
        Histogram thread_word_histogram;

        for (int k=0;k< max_iter;k++) {
#pragma omp for nowait schedule(dynamic,1)
            for (int i = 0; i < doc_count; i++) {
                std::string document_path ="data/Texts/" + std::to_string(i) + ".txt";

                if (std::ifstream file(document_path); file.is_open()) {

                    std::stringstream buffer;
                    buffer << file.rdbuf();
                    std::string content = buffer.str();
                    texts.push_back(content);

                    file.close();
                } else {
                    printf("Impossible open the file: %s", document_path.c_str());
                }
            }
        }
        for (size_t l = 0; l < texts.size(); l++) {
            UpdateHistogramWord(thread_word_histogram, texts[l], n_gram_size);
        }

#pragma omp critical (word)
        {
            for (const auto & [fst, snd]: thread_word_histogram) {
                hist[fst] += snd;
            }
        }

    }
}

Histogram count_par_document_level_tls(const std::string& directory_path, int ngram_size, int num_threads, int multiplier) {
    std::vector<std::vector<std::string>> base_docs;

    for (const auto& entry : fs::directory_iterator(directory_path)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            std::ifstream file(entry.path());
            if (!file.is_open()) continue;

            std::stringstream buffer;
            buffer << file.rdbuf();
            base_docs.push_back(tokenize_text(buffer.str()));
        }
    }

    if (base_docs.empty()) return {};

    std::vector<std::vector<std::string>> scaled_docs;
    scaled_docs.reserve(base_docs.size() * multiplier);

    for (int m = 0; m < multiplier; ++m) {
        for (const auto& doc : base_docs) {
            scaled_docs.push_back(doc);
        }
    }

    std::vector<Histogram> local_hists(num_threads);

    #pragma omp parallel num_threads(num_threads) default(none) \
        shared(scaled_docs, ngram_size, local_hists)
    {
        int tid = omp_get_thread_num();
        Histogram& my_hist = local_hists[tid];

        #pragma omp for schedule(dynamic)
        for (const auto & words : scaled_docs) {
            if (words.size() < static_cast<size_t>(ngram_size)) continue;

            const size_t limit = words.size() - ngram_size;
            for (size_t k = 0; k <= limit; ++k) {
                std::string n_gram = words[k];
                for (int j = 1; j < ngram_size; ++j) {
                    n_gram += " " + words[k + j];
                }
                my_hist[n_gram]++;
            }
        }
    }

    Histogram final_hist;
    size_t total_unique = 0;
    for (const auto& h : local_hists) total_unique += h.size();
    final_hist.reserve(total_unique / 2);

    for (const auto& h : local_hists) {
        for (const auto& [key, val] : h) {
            final_hist[key] += val;
        }
    }

    return final_hist;
}

Histogram count_par_fine_grained_locking(const std::string& directory_path, int n_gram_size, int num_threads, int multiplier)
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

void UpdateHistogramWord(Histogram& hist, const std::string& text, const int n_gram_size) {
    std::istringstream stream(text);

    if (const std::vector<std::string> words((std::istream_iterator<std::string>(stream)), {}); words.size() > static_cast<size_t>(n_gram_size)) {
        for (size_t i = 0; i <= words.size() - static_cast<size_t>(n_gram_size); ++i) {
            std::string word_string;
            for (size_t j = i; j < i + static_cast<size_t>(n_gram_size) - 1; ++j) {
                word_string += words[j] + " ";
            }
            word_string += words[i + n_gram_size - 1];

            for (char &ch: word_string) ch = std::tolower(ch);

            word_string.erase(std::remove_if(word_string.begin(), word_string.end(),
                                             [](const char ch) { return !std::isalpha(ch) && ch != ' '; }),
                              word_string.end());

            hist[word_string]++;
        }
    }
}

