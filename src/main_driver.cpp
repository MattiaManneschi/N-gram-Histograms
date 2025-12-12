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
#include <iomanip>

// TODO OTTIMIZZARE PARALLELIZZAZIONE BASATA SU LOCKS (SE POSSIBILE)
    // TODO PROVARE AD AGGIUNGERE FINE-GRAINED LOCKS O OPERAZIONI ATOMICHE (CHATGPT)
// TODO CREARE UNA FUNZIONE DI PLOT E SALVARE I RISULTATI IN UNA CARTELLA
// TODO CREARE ESEGUIBILI


template <typename CorpusType>
void run_parallel_test(std::chrono::duration<double> duration_seq, const CorpusType &corpus, int n_gram_size, const std::string &strategy_name, int max_threads);
void run_workload_scaling_test(const std::string& data_dir, int n_gram_size, int fixed_threads, const std::vector<int>& multiplier_steps, const std::string& strategy_name);

int main(int argc, char* argv[]) {

    const std::string DATA_DIR = argv[1];

    int n_gram_size = std::stoi(argv[2]);

    const int max_threads = std::stoi(argv[3]);

    std::string test_mode = argv[4];

    std::transform(test_mode.begin(), test_mode.end(), test_mode.begin(), ::toupper);

    auto words = load_and_tokenize_directory(DATA_DIR);
    DocumentCorpus doc_words = load_and_tokenize_document_corpus(DATA_DIR);
    if (words.empty()) return 1;
    
    std::cout << "\nAnalisi " << n_gram_size << "-grammi" << std::endl;

    std::cout << "\nDati caricati da: " << DATA_DIR << std::endl;

    auto start_seq = std::chrono::high_resolution_clock::now();
    Histogram hist_seq = count_sequential(words, n_gram_size);
    auto end_seq = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_seq = end_seq - start_seq;
    std::cout << "\n==============================================" << std::endl;
    std::cout << "\n--- Test Sequenziale ---" << std::endl;
    std::cout << "Tempo: " << duration_seq.count() << " secondi." << std::endl;

    print_corpus_statistics(hist_seq, n_gram_size, duration_seq.count());

    std::cout << "\n--- Test Parallelo ---" << std::endl;

    if (test_mode == "SCALING"){
        std::cout << "\n==============================================" << std::endl;
        std::cout << "Esecuzione Test: Strong Scaling (Load Fixed)" << std::endl;
        std::cout << "==============================================" << std::endl;

        run_parallel_test(duration_seq, words, n_gram_size, "Chunk-based", max_threads);
        run_parallel_test(duration_seq, doc_words, n_gram_size, "Document-level", max_threads);
        run_parallel_test(duration_seq, words, n_gram_size, "Critical-based", max_threads);

    } else if (test_mode == "WORKLOAD"){
        std::cout << "\n==============================================" << std::endl;
        std::cout << "Esecuzione Test: Workload Scaling (Threads Fixed)" << std::endl;
        std::cout << "==============================================" << std::endl;
        
        const int FIXED_THREAD = max_threads;
        const std::vector<int> MULTIPLIER_STEPS = {1, 3, 5, 8, 10};
        
        run_workload_scaling_test(DATA_DIR, n_gram_size, FIXED_THREAD, MULTIPLIER_STEPS, "Chunk-based");
        run_workload_scaling_test(DATA_DIR, n_gram_size, FIXED_THREAD, MULTIPLIER_STEPS, "Document-level");
        run_workload_scaling_test(DATA_DIR, n_gram_size, FIXED_THREAD, MULTIPLIER_STEPS, "Critical-based");
    }

    return 0;
}

template <typename CorpusType>
void run_parallel_test(
    std::chrono::duration<double> duration_seq, 
    const CorpusType& corpus, 
    int n_gram_size, 
    const std::string& strategy_name, 
    int max_threads) 
{
    for (int num_threads = 1; num_threads <= max_threads; ++num_threads){

        Histogram hist_par;

        auto start_par = std::chrono::high_resolution_clock::now();
        
        // Caso 1: Chunk-based
        if constexpr (std::is_same_v<CorpusType, std::vector<std::string>>) {
            if (strategy_name.find("Chunk-based") != std::string::npos){
                hist_par = count_parallel(corpus, n_gram_size, num_threads);
            } else if (strategy_name.find("Critical-based") != std::string::npos){
                // Caso 3: Critical based 
                hist_par = count_parallel_critical_based(corpus, n_gram_size, num_threads);
            }
                } 
        // Caso 2: Document-level
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

void run_workload_scaling_test(
    const std::string &data_dir,
    int n_gram_size,
    int fixed_threads,
    const std::vector<int> &multiplier_steps,
    const std::string &strategy_name)
{
    for (int multiplier : multiplier_steps) {
    
        std::vector<std::string> chunk_words = load_and_tokenize_directory(data_dir, multiplier); 
        if (chunk_words.empty()) continue; 
        
        DocumentCorpus doc_words = load_and_tokenize_document_corpus(data_dir, multiplier);
        
        auto start_seq = std::chrono::high_resolution_clock::now();
        count_sequential(chunk_words, n_gram_size); 
        auto end_seq = std::chrono::high_resolution_clock::now();
        double t_seq = std::chrono::duration_cast<std::chrono::duration<double>>(end_seq - start_seq).count();

        auto start_par = std::chrono::high_resolution_clock::now();
        
        if (strategy_name.find("Chunk-based") != std::string::npos) {
            count_parallel(chunk_words, n_gram_size, fixed_threads);
        } else if (strategy_name.find("Document-level") != std::string::npos) {
            count_parallel_document_level(doc_words, n_gram_size, fixed_threads);
        } else if (strategy_name.find("Critical-based") != std::string::npos) {
            count_parallel_critical_based(chunk_words, n_gram_size, fixed_threads);
        }
        else{return;}

        auto end_par = std::chrono::high_resolution_clock::now();
        double t_par = std::chrono::duration_cast<std::chrono::duration<double>>(end_par - start_par).count();

        double speedup = t_seq / t_par;
        double efficiency = speedup / fixed_threads * 100.0;

        std::cout << std::fixed << std::setprecision(4);
        
        std::cout << "P: " << strategy_name
                  << " | M: " << multiplier
                  << " | T_seq: " << t_seq << "s"
                  << " | T_par: " << t_par << "s"
                  << " | Speedup: " << std::setprecision(3) << speedup
                  << " | Efficienza: " << std::setprecision(3) << (efficiency) << "%" 
                  << std::endl;

        using std::swap; 

        if (!chunk_words.empty()) {
            std::vector<std::string> empty_vec;
            swap(chunk_words, empty_vec);
        }
        if (!doc_words.empty()) {
            for (auto& doc : doc_words) {
                std::vector<std::string> empty_doc_vec;
                swap(doc, empty_doc_vec);
            }
            DocumentCorpus empty_corpus;
            swap(doc_words, empty_corpus);
        }
    }
}
