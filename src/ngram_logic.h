#ifndef NGRAM_LOGIC_H
#define NGRAM_LOGIC_H

#include <string>
#include <vector>
#include <unordered_map>

using Histogram = std::unordered_map<std::string, long long>; 

/**
 * Versione sequenziale dell'algoritmo di conteggio N-grammi.
 * Serve come baseline di performance.
 */
Histogram count_sequential(const std::vector<std::string>& words, int n_gram_size);

/**
 * Versione parallela dell'algoritmo di conteggio N-grammi (implementata con OpenMP).
 * Utilizzerà la strategia Thread-Local per alta scalabilità.
 */
Histogram count_parallel(const std::vector<std::string>& words, int n_gram_size, int num_threads);

#endif // NGRAM_LOGIC_H