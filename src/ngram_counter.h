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
Histogram count_par_chunk_based_tls(const std::string& directory_path, int n_gram_size, int num_threads);

Histogram count_par_document_level_tls(const std::string& directory_path, int ngram_size, int num_threads);

//Parallele locks
Histogram count_par_dynamic_locking(const std::string& directory_path, int n_gram_size, int num_threads);


#endif