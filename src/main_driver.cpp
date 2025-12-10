#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include <omp.h>
#include <cassert>
#include "text_processor.h"
#include "ngram_logic.h"
#include <type_traits>

const std::string DATA_DIR = "data/Texts";

template <typename CorpusType>
void run_parallel_test_suite(std::chrono::duration<double> duration_seq, const CorpusType &corpus, int n_gram_size, const std::string &strategy_name, int max_threads);

int main(int argc, char* argv[]) {
    int ngram_size = std::stoi(argv[2]);
    int max_threads = std::stoi(argv[3]);

    auto words = load_and_tokenize_directory(DATA_DIR);
    DocumentCorpus doc_words = load_and_tokenize_document_corpus(DATA_DIR);
    if (words.empty()) return 1;
    
    std::cout << "\nAnalisi " << ngram_size << "-grammi." << std::endl;

    auto start_seq = std::chrono::high_resolution_clock::now();
    Histogram hist_seq = count_sequential(words, ngram_size);
    auto end_seq = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_seq = end_seq - start_seq;
    std::cout << "\n--- Test Sequenziale ---" << std::endl;
    std::cout << "Tempo: " << duration_seq.count() << " secondi." << std::endl;
    std::cout << "N-grammi unici trovati: " << hist_seq.size() << std::endl;

    std::cout << "\n--- Test Parallelo ---" << std::endl;

    run_parallel_test_suite(duration_seq, words, ngram_size, "Chunk-based", max_threads);
    run_parallel_test_suite(duration_seq, doc_words, ngram_size, "Document-level", max_threads);

    // TODO ADD MULTIPLE PARALLELIZATION VERSION AND COMPARE THEM
    // TODO ADD OPPOSITE TEST (FIXED THREAD NUMBER AND CHANGE DATA SIZE)
    // TODO CREATE A PLOT FUNCTION
    // TODO CREATE EXECUTABLES
    
    return 0;
}

template <typename CorpusType>
void run_parallel_test_suite(std::chrono::duration<double> duration_seq, const CorpusType& corpus, int n_gram_size, const std::string& strategy_name, int max_threads) {
    
    for (int num_threads = 1; num_threads <= max_threads; ++num_threads){

        Histogram hist_par;

        auto start_par = std::chrono::high_resolution_clock::now();
        
        // Caso 1: Chunk-based (Corpus = std::vector<std::string>)
        if constexpr (std::is_same_v<CorpusType, std::vector<std::string>>) {
            hist_par = count_parallel(corpus, n_gram_size, num_threads);
            
        } 
        // Caso 2: Document-level (Corpus = std::vector<std::vector<std::string>>)
        else if constexpr (std::is_same_v<CorpusType, DocumentCorpus>) {
            hist_par = count_parallel_document_level(corpus, n_gram_size, num_threads);
        } else {
            std::cerr << "Errore: Tipo di corpus non riconosciuto." << std::endl;
            return;
        }

        auto end_par = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration_par = end_par - start_par;
        double speedup = duration_seq.count() / duration_par.count();
        double efficiency = speedup / num_threads;

        std::cout << "P: " << strategy_name
                  << " | T: " << num_threads 
                  << " | Tempo: " << duration_par.count() << "s"
                  << " | Speedup: " << speedup 
                  << " | Efficienza: " << (efficiency * 100.0) << "%" << std::endl;
    }
}
