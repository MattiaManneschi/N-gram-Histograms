/**
 * =============================================================================
 * statistics.cpp - N-gram Statistics Implementation
 * =============================================================================
 */

#include "statistics.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <climits>
#include <numeric>


HistogramStats compute_histogram_stats(const Histogram& hist, int top_k_count)
{
    HistogramStats stats{};

    if (hist.empty())
    {
        stats.total_ngrams = 0;
        stats.unique_ngrams = 0;
        stats.max_frequency = 0;
        stats.min_frequency = 0;
        stats.mean_frequency = 0;
        stats.std_deviation = 0;
        return stats;
    }

    stats.unique_ngrams = hist.size();
    stats.total_ngrams = 0;
    stats.max_frequency = 0;
    stats.min_frequency = INT_MAX;


    for (const auto& [ngram, count] : hist)
    {
        stats.total_ngrams += count;

        if (count > stats.max_frequency)
        {
            stats.max_frequency = count;
            stats.most_common = ngram;
        }
        if (count < stats.min_frequency)
        {
            stats.min_frequency = count;
            stats.least_common = ngram;
        }
    }


    stats.mean_frequency = static_cast<double>(stats.total_ngrams) / stats.unique_ngrams;


    double variance_sum = 0.0;
    for (const auto& [ngram, count] : hist)
    {
        double diff = count - stats.mean_frequency;
        variance_sum += diff * diff;
    }
    stats.std_deviation = std::sqrt(variance_sum / stats.unique_ngrams);


    std::vector<std::pair<std::string, int>> all_entries(hist.begin(), hist.end());
    int k = std::min(top_k_count, static_cast<int>(all_entries.size()));

    std::partial_sort(all_entries.begin(), all_entries.begin() + k, all_entries.end(),
                      [](const auto& a, const auto& b) { return a.second > b.second; });

    stats.top_k.assign(all_entries.begin(), all_entries.begin() + k);

    return stats;
}

FrequencyDistribution compute_frequency_distribution(const Histogram& hist)
{
    FrequencyDistribution dist{};

    if (hist.empty())
    {
        dist.num_hapax_legomena = 0;
        dist.num_dis_legomena = 0;
        dist.num_tris_legomena = 0;
        dist.hapax_ratio = 0;
        dist.coverage_top_100 = 0;
        return dist;
    }


    for (const auto& [ngram, count] : hist)
    {
        dist.distribution[count]++;
    }


    auto it1 = dist.distribution.find(1);
    dist.num_hapax_legomena = (it1 != dist.distribution.end()) ? it1->second : 0;


    auto it2 = dist.distribution.find(2);
    dist.num_dis_legomena = (it2 != dist.distribution.end()) ? it2->second : 0;


    auto it3 = dist.distribution.find(3);
    dist.num_tris_legomena = (it3 != dist.distribution.end()) ? it3->second : 0;


    dist.hapax_ratio = static_cast<double>(dist.num_hapax_legomena) / hist.size();


    std::vector<int> frequencies;
    frequencies.reserve(hist.size());
    for (const auto& [ngram, count] : hist)
    {
        frequencies.push_back(count);
    }

    std::partial_sort(frequencies.begin(),
                      frequencies.begin() + std::min(100, static_cast<int>(frequencies.size())),
                      frequencies.end(),
                      std::greater<int>());

    long long top_100_sum = 0;
    long long total_sum = 0;
    for (size_t i = 0; i < frequencies.size(); ++i)
    {
        total_sum += frequencies[i];
        if (i < 100) top_100_sum += frequencies[i];
    }

    dist.coverage_top_100 = (total_sum > 0) ? static_cast<double>(top_100_sum) / total_sum * 100.0 : 0.0;

    return dist;
}


void print_histogram_stats(const HistogramStats& stats)
{
    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "                   HISTOGRAM STATISTICS                      \n";
    std::cout << "============================================================\n";
    std::cout << "  Total n-gram tokens:     " << std::setw(15) << stats.total_ngrams << "\n";
    std::cout << "  Unique n-grams (vocab):  " << std::setw(15) << stats.unique_ngrams << "\n";
    std::cout << "  Max frequency:           " << std::setw(15) << stats.max_frequency << "\n";
    std::cout << "  Min frequency:           " << std::setw(15) << stats.min_frequency << "\n";
    std::cout << "  Mean frequency:          " << std::setw(15) << std::fixed
        << std::setprecision(2) << stats.mean_frequency << "\n";
    std::cout << "  Std deviation:           " << std::setw(15) << std::setprecision(2)
        << stats.std_deviation << "\n";
    std::cout << "------------------------------------------------------------\n";
    std::cout << "  Most common: \"" << stats.most_common << "\" ("
        << stats.max_frequency << " occurrences)\n";
    std::cout << "============================================================\n";

    if (!stats.top_k.empty())
    {
        std::cout << "\n  TOP " << stats.top_k.size() << " N-GRAMS:\n";
        std::cout << "  ----------------------------------------------------------\n";
        int rank = 1;
        for (const auto& [ngram, count] : stats.top_k)
        {
            std::cout << "  " << std::setw(3) << rank++ << ". "
                << std::setw(30) << std::left << ("\"" + ngram + "\"")
                << std::right << std::setw(12) << count << "\n";
        }
    }
}

void print_frequency_distribution(const FrequencyDistribution& dist)
{
    std::cout << "\n";
    std::cout << "============================================================\n";
    std::cout << "              FREQUENCY DISTRIBUTION (Zipf Analysis)         \n";
    std::cout << "============================================================\n";
    std::cout << "  Hapax legomena (freq=1):   " << std::setw(12) << dist.num_hapax_legomena << "\n";
    std::cout << "  Dis legomena (freq=2):     " << std::setw(12) << dist.num_dis_legomena << "\n";
    std::cout << "  Tris legomena (freq=3):    " << std::setw(12) << dist.num_tris_legomena << "\n";
    std::cout << "  Hapax ratio:               " << std::setw(12) << std::fixed
        << std::setprecision(1) << (dist.hapax_ratio * 100) << "%\n";
    std::cout << "  Coverage by top 100:       " << std::setw(12) << std::setprecision(1)
        << dist.coverage_top_100 << "%\n";
    std::cout << "------------------------------------------------------------\n";


    std::vector<std::pair<int, int>> sorted_dist(dist.distribution.begin(),
                                                 dist.distribution.end());
    std::sort(sorted_dist.begin(), sorted_dist.end());

    std::cout << "  Frequency -> Count (first 10):\n";
    int shown = 0;
    for (const auto& [freq, count] : sorted_dist)
    {
        if (shown++ >= 10) break;
        std::cout << "    freq=" << std::setw(6) << freq
            << " -> " << std::setw(10) << count << " n-grams\n";
    }
    std::cout << "============================================================\n";
}

void print_full_statistics(const FullStatistics& stats)
{
    std::cout << "\n";
    std::cout << "############################################################\n";
    std::cout << "#                 FULL STATISTICS REPORT                   #\n";
    std::cout << "############################################################\n";
    std::cout << "  Strategy:        " << stats.strategy_name << "\n";
    std::cout << "  N-gram size:     " << stats.ngram_size << "\n";
    std::cout << "  Threads:         " << stats.num_threads << "\n";
    std::cout << "  Multiplier:      " << stats.workload_multiplier << "\n";
    std::cout << "  Execution time:  " << std::fixed << std::setprecision(3)
        << stats.execution_time_sec << "s\n";
    std::cout << "  Speedup:         " << std::setprecision(2) << stats.speedup << "x\n";
    std::cout << "  Efficiency:      " << std::setprecision(1)
        << (stats.efficiency * 100) << "%\n";
    std::cout << "############################################################\n";

    print_histogram_stats(stats.histogram);
    print_frequency_distribution(stats.frequency);
}


void export_stats_csv(const FullStatistics& stats, const std::string& filepath)
{
    std::ofstream file(filepath);
    if (!file.is_open())
    {
        std::cerr << "ERROR: Cannot open " << filepath << std::endl;
        return;
    }

    file << "metric,value\n";
    file << "strategy," << stats.strategy_name << "\n";
    file << "ngram_size," << stats.ngram_size << "\n";
    file << "num_threads," << stats.num_threads << "\n";
    file << "multiplier," << stats.workload_multiplier << "\n";
    file << "execution_time_sec," << std::fixed << std::setprecision(6)
        << stats.execution_time_sec << "\n";
    file << "speedup," << std::setprecision(4) << stats.speedup << "\n";
    file << "efficiency," << stats.efficiency << "\n";
    file << "total_ngrams," << stats.histogram.total_ngrams << "\n";
    file << "unique_ngrams," << stats.histogram.unique_ngrams << "\n";
    file << "max_frequency," << stats.histogram.max_frequency << "\n";
    file << "min_frequency," << stats.histogram.min_frequency << "\n";
    file << "mean_frequency," << std::setprecision(4) << stats.histogram.mean_frequency << "\n";
    file << "std_deviation," << stats.histogram.std_deviation << "\n";
    file << "most_common,\"" << stats.histogram.most_common << "\"\n";
    file << "hapax_legomena," << stats.frequency.num_hapax_legomena << "\n";
    file << "dis_legomena," << stats.frequency.num_dis_legomena << "\n";
    file << "tris_legomena," << stats.frequency.num_tris_legomena << "\n";
    file << "hapax_ratio," << stats.frequency.hapax_ratio << "\n";
    file << "coverage_top_100," << stats.frequency.coverage_top_100 << "\n";

    file.close();
    std::cout << "Exported: " << filepath << std::endl;
}

void export_top_k_csv(const std::vector<std::pair<std::string, int>>& top_k,
                      const std::string& filepath)
{
    std::ofstream file(filepath);
    if (!file.is_open())
    {
        std::cerr << "ERROR: Cannot open " << filepath << std::endl;
        return;
    }

    file << "rank,ngram,frequency\n";
    int rank = 1;
    for (const auto& [ngram, count] : top_k)
    {
        std::string escaped = ngram;
        size_t pos = 0;
        while ((pos = escaped.find('"', pos)) != std::string::npos)
        {
            escaped.replace(pos, 1, "\"\"");
            pos += 2;
        }
        file << rank++ << ",\"" << escaped << "\"," << count << "\n";
    }

    file.close();
    std::cout << "Exported: " << filepath << std::endl;
}

void export_frequency_distribution_csv(const FrequencyDistribution& dist,
                                       const std::string& filepath)
{
    std::ofstream file(filepath);
    if (!file.is_open())
    {
        std::cerr << "ERROR: Cannot open " << filepath << std::endl;
        return;
    }


    std::vector<std::pair<int, int>> sorted(dist.distribution.begin(),
                                            dist.distribution.end());
    std::sort(sorted.begin(), sorted.end());

    file << "frequency,count\n";
    for (const auto& [freq, count] : sorted)
    {
        file << freq << "," << count << "\n";
    }

    file.close();
    std::cout << "Exported: " << filepath << std::endl;
}

void export_full_report(const FullStatistics& stats, const std::string& filepath)
{
    std::ofstream file(filepath);
    if (!file.is_open())
    {
        std::cerr << "ERROR: Cannot open " << filepath << std::endl;
        return;
    }

    file << "================================================================\n";
    file << "              N-GRAM ANALYSIS FULL REPORT                       \n";
    file << "================================================================\n\n";

    file << "CONFIGURATION\n";
    file << "----------------------------------------------------------------\n";
    file << "  Strategy:           " << stats.strategy_name << "\n";
    file << "  N-gram size:        " << stats.ngram_size << "\n";
    file << "  Threads:            " << stats.num_threads << "\n";
    file << "  Workload multiplier:" << stats.workload_multiplier << "\n\n";

    file << "PERFORMANCE\n";
    file << "----------------------------------------------------------------\n";
    file << "  Execution time:     " << std::fixed << std::setprecision(3)
        << stats.execution_time_sec << " seconds\n";
    file << "  Speedup:            " << std::setprecision(2) << stats.speedup << "x\n";
    file << "  Efficiency:         " << std::setprecision(1)
        << (stats.efficiency * 100) << "%\n\n";

    file << "HISTOGRAM STATISTICS\n";
    file << "----------------------------------------------------------------\n";
    file << "  Total n-gram tokens:    " << stats.histogram.total_ngrams << "\n";
    file << "  Unique n-grams (vocab): " << stats.histogram.unique_ngrams << "\n";
    file << "  Max frequency:          " << stats.histogram.max_frequency << "\n";
    file << "  Min frequency:          " << stats.histogram.min_frequency << "\n";
    file << "  Mean frequency:         " << std::setprecision(2)
        << stats.histogram.mean_frequency << "\n";
    file << "  Std deviation:          " << stats.histogram.std_deviation << "\n";
    file << "  Most common n-gram:     \"" << stats.histogram.most_common << "\"\n\n";

    file << "TOP " << stats.histogram.top_k.size() << " N-GRAMS\n";
    file << "----------------------------------------------------------------\n";
    int rank = 1;
    for (const auto& [ngram, count] : stats.histogram.top_k)
    {
        file << "  " << std::setw(3) << rank++ << ". \"" << ngram
            << "\" - " << count << "\n";
    }
    file << "\n";

    file << "FREQUENCY DISTRIBUTION (Zipf's Law Analysis)\n";
    file << "----------------------------------------------------------------\n";
    file << "  Hapax legomena (freq=1): " << stats.frequency.num_hapax_legomena << "\n";
    file << "  Dis legomena (freq=2):   " << stats.frequency.num_dis_legomena << "\n";
    file << "  Tris legomena (freq=3):  " << stats.frequency.num_tris_legomena << "\n";
    file << "  Hapax ratio:             " << std::setprecision(1)
        << (stats.frequency.hapax_ratio * 100) << "%\n";
    file << "  Coverage by top 100:     " << stats.frequency.coverage_top_100 << "%\n";
    file << "\n";

    file << "================================================================\n";
    file << "  This distribution follows Zipf's Law if hapax ratio is high\n";
    file << "  (typically 40-60% for natural language text)\n";
    file << "================================================================\n";

    file.close();
    std::cout << "Exported: " << filepath << std::endl;
}
