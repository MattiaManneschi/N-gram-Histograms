#ifndef MID_TERM_NGRAM_PARALLEL_RESULTS_EXPORTS_H
#define MID_TERM_NGRAM_PARALLEL_RESULTS_EXPORTS_H

#include <string>
#include <vector>
#include <fstream>

struct BenchmarkResult {
    std::string strategy_name;
    int num_threads;
    double time_seconds;
    double speedup;
    double efficiency;
    int workload_multiplier;  // Per workload scaling (1 se non applicabile)
};

class ResultsExporter {
private:
    std::string output_dir;
    std::vector<BenchmarkResult> results;

public:
    explicit ResultsExporter(const std::string& dir = "results");

    void add_result(const std::string& strategy, int threads, double time,
                   double speedup, double efficiency, int multiplier = 1);

    void save_scaling_results(const std::string& filename, int ngram_size) const;

    void save_workload_results(const std::string& filename, int ngram_size, int fixed_threads) const;

    void clear();

    void save_summary(const std::string& filename, int ngram_size) const;
};



#endif //MID_TERM_NGRAM_PARALLEL_RESULTS_EXPORTS_H
