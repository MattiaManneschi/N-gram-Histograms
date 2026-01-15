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

// Parallelo Thread scaling

void count_par_singleReader_Worker_TLS(int n_gram_size, int max_iter)
{
     std::vector<std::string> texts;

    Histogram hist;

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

void count_par_onTheFly_parallelIO(const int n_gram_size, int max_iter){

    // Count the number of texts in the folder
    int doc_count = 0;
    for ([[maybe_unused]] const auto &document: std::filesystem::directory_iterator("data/Texts")) {
        doc_count++;
    }

    Histogram hist;

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

void count_par_hybrid_preload_TLS(int n_gram_size, int max_iter){

    int doc_count = 0;
    for ([[maybe_unused]] const auto &document: std::filesystem::directory_iterator("data/Texts")) {
        doc_count++;
    }

    Histogram hist;

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

// Parallelo Workload scaling

void count_par_document_level_tls(const std::string& directory_path, int ngram_size, int num_threads, int multiplier)
{
    // FASE 1: Raccogli path file
    std::vector<fs::path> file_paths;
    for (const auto& entry : fs::directory_iterator(directory_path)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            file_paths.push_back(entry.path());
        }
    }

    const size_t num_docs = file_paths.size();
    const size_t total_tasks = num_docs * static_cast<size_t>(multiplier);

    // FASE 2: Caricamento PARALLELO (era sequenziale!)
    std::vector<std::vector<std::string>> tokenized_docs(num_docs);

    #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for (size_t i = 0; i < num_docs; ++i) {
        std::string content = read_file_fast(file_paths[i]);
        tokenized_docs[i] = tokenize_text(content);
    }

    // FASE 3: Pre-allocazione histogram
    std::vector<Histogram> local_hists(num_threads);

    size_t estimated_ngrams = 0;
    for (const auto& doc : tokenized_docs) {
        if (doc.size() >= static_cast<size_t>(ngram_size)) {
            estimated_ngrams += doc.size() - ngram_size + 1;
        }
    }
    estimated_ngrams *= multiplier;

    for (auto& hist : local_hists) {
        hist.reserve(estimated_ngrams / (num_threads * 3));
    }

    // FASE 4: Conteggio ZERO-COPY
    #pragma omp parallel num_threads(num_threads)
    {
        const int tid = omp_get_thread_num();
        Histogram& my_hist = local_hists[tid];

        // Buffer riutilizzabile (1 allocazione invece di milioni)
        std::string ngram_buffer;
        ngram_buffer.reserve(128);

        #pragma omp for schedule(dynamic, 4)
        for (size_t task_id = 0; task_id < total_tasks; ++task_id) {

            // ZERO-COPY: accesso tramite indice, nessuna copia
            const size_t doc_idx = task_id % num_docs;
            const std::vector<std::string>& words = tokenized_docs[doc_idx];

            if (words.size() < static_cast<size_t>(ngram_size)) continue;

            const size_t limit = words.size() - ngram_size;

            for (size_t k = 0; k <= limit; ++k) {
                build_ngram_inplace(words, k, ngram_size, ngram_buffer);
                my_hist[ngram_buffer]++;
            }
        }
    }

    // FASE 5: Riduzione ottimizzata (move invece di copia)
    size_t max_idx = 0;
    size_t max_size = 0;
    for (size_t i = 0; i < local_hists.size(); ++i) {
        if (local_hists[i].size() > max_size) {
            max_size = local_hists[i].size();
            max_idx = i;
        }
    }

    Histogram final_hist = std::move(local_hists[max_idx]);

    for (size_t i = 0; i < local_hists.size(); ++i) {
        if (i == max_idx) continue;
        for (const auto& [ngram, count] : local_hists[i]) {
            final_hist[ngram] += count;
        }
    }
}

void count_par_fine_grained_locking(const std::string& directory_path, int n_gram_size, int num_threads, int multiplier)
{
    std::vector<fs::path> file_paths;
    fs::path dir_path(directory_path);

    for (const auto& entry : fs::directory_iterator(dir_path)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            file_paths.push_back(entry.path());
        }
    }

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
}

void count_par_chunk_based_adaptive(const std::string& directory_path, int ngram_size, int num_threads, int multiplier){
    // -----------------------------------------------------------------------
    // FASE 1: Carica corpus
    // -----------------------------------------------------------------------
    std::vector<fs::path> file_paths;
    for (const auto& entry : fs::directory_iterator(directory_path)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            file_paths.push_back(entry.path());
        }
    }

    std::vector<std::vector<std::string>> doc_words(file_paths.size());

    #pragma omp parallel for num_threads(num_threads) schedule(dynamic)
    for (size_t i = 0; i < file_paths.size(); ++i) {
        std::string content = read_file_fast(file_paths[i]);
        doc_words[i] = tokenize_text(content);
    }

    size_t total_words = 0;
    for (const auto& dw : doc_words) {
        total_words += dw.size();
    }

    std::vector<std::string> corpus;
    corpus.reserve(total_words * multiplier);

    for (int m = 0; m < multiplier; ++m) {
        for (const auto& dw : doc_words) {
            corpus.insert(corpus.end(), dw.begin(), dw.end());
        }
    }

    doc_words.clear();
    doc_words.shrink_to_fit();

    const size_t corpus_size = corpus.size();

    // -----------------------------------------------------------------------
    // FASE 2: Chunk size ADATTIVO
    //         Obiettivo: circa 4-8 chunk per thread per buon bilanciamento
    // -----------------------------------------------------------------------

    const size_t num_ngrams = corpus_size - ngram_size + 1;
    const size_t target_chunks_per_thread = 6;
    const size_t target_total_chunks = num_threads * target_chunks_per_thread;

    // Chunk size calcolato dinamicamente
    size_t chunk_size = std::max(
        static_cast<size_t>(1000),  // Minimo 1000 parole
        num_ngrams / target_total_chunks
    );

    const size_t overlap = static_cast<size_t>(ngram_size - 1);
    const size_t effective_chunk = chunk_size;
    const size_t num_chunks = (num_ngrams + effective_chunk - 1) / effective_chunk;

    // -----------------------------------------------------------------------
    // FASE 3: Processing
    // -----------------------------------------------------------------------

    std::vector<Histogram> local_hists(num_threads);
    for (auto& h : local_hists) {
        h.reserve(num_ngrams / (num_threads * 3));
    }

    #pragma omp parallel num_threads(num_threads)
    {
        const int tid = omp_get_thread_num();
        Histogram& my_hist = local_hists[tid];

        std::string ngram_buffer;
        ngram_buffer.reserve(128);

        #pragma omp for schedule(dynamic)
        for (size_t chunk_id = 0; chunk_id < num_chunks; ++chunk_id) {

            const size_t start_ngram = chunk_id * effective_chunk;
            const size_t end_ngram = std::min(start_ngram + effective_chunk, num_ngrams);

            for (size_t k = start_ngram; k < end_ngram; ++k) {
                build_ngram_inplace(corpus, k, ngram_size, ngram_buffer);
                my_hist[ngram_buffer]++;
            }
        }
    }

    // -----------------------------------------------------------------------
    // FASE 4: Riduzione
    // -----------------------------------------------------------------------

    size_t max_idx = 0;
    size_t max_size = 0;
    for (size_t i = 0; i < local_hists.size(); ++i) {
        if (local_hists[i].size() > max_size) {
            max_size = local_hists[i].size();
            max_idx = i;
        }
    }

    Histogram final_hist = std::move(local_hists[max_idx]);

    for (size_t i = 0; i < local_hists.size(); ++i) {
        if (i == max_idx) continue;
        for (const auto& [ngram, count] : local_hists[i]) {
            final_hist[ngram] += count;
        }
    }

}

// Support functions

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

static std::string read_file_fast(const fs::path& filepath) {
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return "";

    auto size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::string content(static_cast<size_t>(size), '\0');
    file.read(&content[0], size);

    return content;
}

static void build_ngram_inplace(const std::vector<std::string>& words, size_t start_idx, int ngram_size, std::string& buffer){
    buffer.clear();
    buffer.append(words[start_idx]);

    for (int j = 1; j < ngram_size; ++j) {
        buffer.push_back(' ');
        buffer.append(words[start_idx + j]);
    }
}