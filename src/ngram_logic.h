#ifndef NGRAM_LOGIC_H
#define NGRAM_LOGIC_H

#include <string>
#include <vector>
#include <unordered_map>
#include <omp.h>

using Histogram = std::unordered_map<std::string, long long>; 
using DocumentCorpus = std::vector<std::vector<std::string>>;


Histogram count_sequential(const std::vector<std::string>& words, int n_gram_size);

Histogram count_parallel(const std::vector<std::string>& words, int n_gram_size, int num_threads);

Histogram count_parallel_document_level(const DocumentCorpus& doc_words, int ngram_size, int num_threads);

Histogram count_parallel_critical_based(const std::vector<std::string>& words, int n_gram_size, int requested_threads);

void print_corpus_statistics(const Histogram& hist, int n_gram_size, double total_time);

#endif