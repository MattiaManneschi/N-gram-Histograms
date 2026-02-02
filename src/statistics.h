/**
 * =============================================================================
 * statistics.h - N-gram Statistics Module
 * =============================================================================
 *
 * Modulo per calcolare statistiche e distribuzioni di frequenza sugli histogram.
 * Richiesto dalle linee guida: "Generate frequency distributions and statistics"
 *
 */

#ifndef STATISTICS_H
#define STATISTICS_H

#include <string>
#include <vector>
#include <unordered_map>

using Histogram = std::unordered_map<std::string, int>;


/**
 * Statistiche base dell'histogram
 */
struct HistogramStats
{
    size_t total_ngrams;
    size_t unique_ngrams;
    int max_frequency;
    int min_frequency;
    double mean_frequency;
    double std_deviation;
    std::string most_common;
    std::string least_common;
    std::vector<std::pair<std::string, int>> top_k;
};

/**
 * Distribuzione delle frequenze (per analisi Zipf)
 */
struct FrequencyDistribution
{
    std::unordered_map<int, int> distribution;
    int num_hapax_legomena;
    int num_dis_legomena;
    int num_tris_legomena;
    double hapax_ratio;
    double coverage_top_100;
};

/**
 * Statistiche complete per report
 */
struct FullStatistics
{
    HistogramStats histogram;
    FrequencyDistribution frequency;
    std::string strategy_name;
    int ngram_size;
    int num_threads;
    int workload_multiplier;
    double execution_time_sec;
    double speedup;
    double efficiency;
};


/**
 * Calcola statistiche base dell'histogram
 * @param hist L'histogram da analizzare
 * @param top_k_count Numero di top n-grammi da estrarre (default 20)
 * @return HistogramStats con tutte le statistiche
 */
HistogramStats compute_histogram_stats(const Histogram& hist, int top_k_count = 20);

/**
 * Calcola distribuzione delle frequenze (per analisi Zipf's Law)
 * @param hist L'histogram da analizzare
 * @return FrequencyDistribution con la distribuzione
 */
FrequencyDistribution compute_frequency_distribution(const Histogram& hist);


/**
 * Stampa statistiche histogram su console
 */
void print_histogram_stats(const HistogramStats& stats);

/**
 * Stampa distribuzione frequenze su console
 */
void print_frequency_distribution(const FrequencyDistribution& dist);

/**
 * Stampa report completo
 */
void print_full_statistics(const FullStatistics& stats);


/**
 * Esporta statistiche in formato CSV
 * @param stats Statistiche da esportare
 * @param filepath Path del file di output
 */
void export_stats_csv(const FullStatistics& stats, const std::string& filepath);

/**
 * Esporta top-K n-grammi in CSV
 * @param top_k Vettore di (ngram, frequency)
 * @param filepath Path del file di output
 */
void export_top_k_csv(const std::vector<std::pair<std::string, int>>& top_k,
                      const std::string& filepath);

/**
 * Esporta distribuzione frequenze in CSV (per plot Zipf)
 * @param dist Distribuzione da esportare
 * @param filepath Path del file di output
 */
void export_frequency_distribution_csv(const FrequencyDistribution& dist,
                                       const std::string& filepath);

/**
 * Esporta report completo in formato testo
 * @param stats Statistiche complete
 * @param filepath Path del file di output
 */
void export_full_report(const FullStatistics& stats, const std::string& filepath);

#endif
