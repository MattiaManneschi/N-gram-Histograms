#include "results_exports.h"
#include <iostream>
#include <iomanip>
#include <filesystem>
#include <algorithm>
#include <fstream>

namespace fs = std::filesystem;

ResultsExporter::ResultsExporter(const std::string& dir) : output_dir(dir) {
    // Crea la directory se non esiste
    if (!fs::exists(output_dir)) {
        fs::create_directories(output_dir);
        std::cout << "Creata directory: " << output_dir << std::endl;
    }
}

void ResultsExporter::add_result(const std::string& strategy, const int threads,
                                 const double time, const double speedup, const double efficiency,
                                 const int multiplier) {
    BenchmarkResult result;
    result.strategy_name = strategy;
    result.num_threads = threads;
    result.time_seconds = time;
    result.speedup = speedup;
    result.efficiency = efficiency;
    result.workload_multiplier = multiplier;
    results.push_back(result);
}

void ResultsExporter::save_scaling_results(const std::string& filename, int ngram_size) const
{
    const std::string filepath = output_dir + "/" + filename;
    std::ofstream file(filepath);
    
    if (!file.is_open()) {
        std::cerr << "Errore: impossibile aprire " << filepath << std::endl;
        return;
    }
    
    // Header CSV
    file << "Strategy,Threads,Time_seconds,Speedup,Efficiency_percent\n";
    
    // Dati
    file << std::fixed << std::setprecision(6);
    for (const auto& r : results) {
        file << r.strategy_name << ","
             << r.num_threads << ","
             << r.time_seconds << ","
             << r.speedup << ","
             << (r.efficiency * 100.0) << "\n";
    }
    
    file.close();
    std::cout << "\n✓ Risultati salvati in: " << filepath << std::endl;
}

void ResultsExporter::save_workload_results(const std::string& filename, 
                                           int ngram_size, int fixed_threads) const
{
    const std::string filepath = output_dir + "/" + filename;
    std::ofstream file(filepath);
    
    if (!file.is_open()) {
        std::cerr << "Errore: impossibile aprire " << filepath << std::endl;
        return;
    }
    
    // Header CSV
    file << "Strategy,Multiplier,Threads,Time_seconds,Speedup,Efficiency_percent\n";
    
    // Dati
    file << std::fixed << std::setprecision(6);
    for (const auto& [strategy_name, num_threads, time_seconds, speedup, efficiency, workload_multiplier] : results) {
        file << strategy_name << ","
             << workload_multiplier << ","
             << num_threads << ","
             << time_seconds << ","
             << speedup << ","
             << (efficiency * 100.0) << "\n";
    }
    
    file.close();
    std::cout << "\n✓ Risultati salvati in: " << filepath << std::endl;
}

void ResultsExporter::save_summary(const std::string& filename, int ngram_size) const
{
    std::string filepath = output_dir + "/" + filename;
    std::ofstream file(filepath);
    
    if (!file.is_open()) {
        std::cerr << "Errore: impossibile aprire " << filepath << std::endl;
        return;
    }
    
    file << "===============================================\n";
    file << "SUMMARY REPORT - " << ngram_size << "-grammi\n";
    file << "===============================================\n\n";
    
    // Raggruppa per strategia
    std::vector<std::string> strategies;
    for (const auto& r : results) {
        if (std::find(strategies.begin(), strategies.end(), r.strategy_name) == strategies.end()) {
            strategies.push_back(r.strategy_name);
        }
    }
    
    for (const auto& strategy : strategies) {
        file << "\n--- " << strategy << " ---\n";
        
        double max_speedup = 0;
        int best_threads = 0;
        
        for (const auto& r : results) {
            if (r.strategy_name == strategy) {
                file << "  Threads: " << std::setw(2) << r.num_threads 
                     << " | Time: " << std::fixed << std::setprecision(4) << r.time_seconds << "s"
                     << " | Speedup: " << std::setprecision(2) << r.speedup
                     << " | Efficiency: " << std::setprecision(1) << (r.efficiency * 100.0) << "%\n";
                
                if (r.speedup > max_speedup) {
                    max_speedup = r.speedup;
                    best_threads = r.num_threads;
                }
            }
        }
        
        file << "  → Best: " << best_threads << " threads (speedup: " 
             << std::setprecision(2) << max_speedup << "x)\n";
    }
    
    file.close();
    std::cout << "✓ Summary salvato in: " << filepath << std::endl;
}

void ResultsExporter::clear() {
    results.clear();
}