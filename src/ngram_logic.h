#ifndef NGRAM_LOGIC_H
#define NGRAM_LOGIC_H

#include <string>
#include <vector>
#include <unordered_map>

using Histogram = std::unordered_map<std::string, long long>; 
using DocumentCorpus = std::vector<std::vector<std::string>>;

//Sequenziale
Histogram count_seq(const std::vector<std::string>& words, int n_gram_size);

//Parallele TLS
Histogram count_par_static_tls(const std::vector<std::string>& words, int n_gram_size, int num_threads);

Histogram count_par_dynamic_tls(const DocumentCorpus& doc_words, int ngram_size, int num_threads);

//Parallele locks
Histogram count_par_coarse_grained(const std::vector<std::string>& words, int n_gram_size, int requested_threads);

Histogram count_par_fine_grained(const std::vector<std::string>& words, int n_gram_size, int num_threads);

void print_corpus_statistics(const Histogram& hist, int n_gram_size, double total_time);

#endif