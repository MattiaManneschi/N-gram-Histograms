#ifndef NGRAM_LOGIC_H
#define NGRAM_LOGIC_H

#include <fstream>
#include <unordered_map>
#include <filesystem>
#include <vector>

using Histogram = std::unordered_map<std::string, int>;

// Sequenziale

Histogram count_seq(const std::vector<std::string>& words, int n_gram_size);

// Parallelo thread
void count_par_hybrid_preload_TLS(int n_gram_size, int max_iter = 1);
void count_par_singleReader_Worker_TLS(int n_gram_size, int max_iter = 1);
void count_par_onTheFly_parallelIO(int n_gram_size, int max_iter = 1);


// Parallelo workload
void count_par_chunk_based_adaptive(const std::string& directory_path, int ngram_size, int num_threads, int multiplier);
void count_par_document_level_tls(const std::string& directory_path, int ngram_size, int num_threads, int multiplier);
void count_par_fine_grained_locking(const std::string& directory_path, int n_gram_size, int num_threads, int multiplier);


void UpdateHistogramWord(Histogram& hist, const std::string& text, int n_gram_size);
static std::string read_file_fast(const std::filesystem::path& filepath);
static void build_ngram_inplace(const std::vector<std::string>& words, size_t start_idx, int ngram_size, std::string& buffer);

#endif
