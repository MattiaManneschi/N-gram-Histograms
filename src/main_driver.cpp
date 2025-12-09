#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <omp.h>
#include <cassert>
#include "text_processor.h"
#include "ngram_logic.h"    



int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Uso: " << argv[0] << " <nome_directory_corpus> <ngram_size>" << std::endl;
        std::cerr << "Esempio: " << argv[0] << " data/ 2" << std::endl;
        return 1;
    }
    std::string dirname = argv[1];
    int ngram_size = std::stoi(argv[2]);

    auto words = load_and_tokenize_directory(dirname); 
    if (words.empty()) return 1;
    std::cout << "Corpus caricato con " << words.size() << " parole. Analisi " << ngram_size << "-grammi." << std::endl;
    

    auto start_seq = std::chrono::high_resolution_clock::now();
    Histogram hist_seq = count_sequential(words, ngram_size);
    auto end_seq = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_seq = end_seq - start_seq;
    std::cout << "\n--- Test Sequenziale ---" << std::endl;
    std::cout << "Tempo: " << duration_seq.count() << " secondi." << std::endl;
    std::cout << "N-grammi unici trovati: " << hist_seq.size() << std::endl;
    

    std::cout << "\n--- Test Parallelo (OpenMP) e Speedup ---" << std::endl;

    int max_threads = 16;

    for (int num_threads = 1; num_threads <= max_threads; ++num_threads) { 

        omp_set_num_threads(num_threads);

        auto start_par = std::chrono::high_resolution_clock::now();
        Histogram hist_par = count_parallel(words, ngram_size, num_threads);
        auto end_par = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration_par = end_par - start_par;
        
        double speedup = duration_seq.count() / duration_par.count();
        double efficiency = speedup / num_threads;
        
        std::cout << "T: " << num_threads 
                  << " | Tempo: " << duration_par.count() << "s"
                  << " | Speedup: " << speedup 
                  << " | Efficienza: " << (efficiency * 100.0) << "%" << std::endl;
    }

    // TODO ADD MULTIPLE PARALLELIZATION VERSION AND COMPARE THEM
    // TODO ADD OPPOSITE TEST (FIXED THREAD NUMBER AND CHANGE DATA SIZE)
    // TODO CREATE A PLOT FUNCTION
    // TODO CREATE EXECUTABLES
    
    return 0;
}