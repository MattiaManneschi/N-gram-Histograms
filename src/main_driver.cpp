#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <algorithm>
#include "text_processor.h"
#include "ngram_logic.h"
#include "results_exports.h"
#include <type_traits>
#include <iomanip>

//TODO OTTIMIZZAZIONE CODICE
//TODO UN SOLO TEST SEQUENZIALE PER THREAD SCALING
//TODO M TEST SEQUENZIALI PER WORKLOAD SCALING (UNO SOLO PER OGNI MULTIPLIER) E POI AGGIUNGERE CONFRONTO
//TODO SPOSTARE ESECUZIONE TEST SEQUENZIALE DIRETTAMENTE DENTRO LE FUNZIONI DI TEST
//TODO DIVIDERE LE FUNZIONI DI TEST PER THREAD SCALING E WORKLOAD SCALING
//TODO RINOMINARE LE FUNZIONI IN TAL SENSO

template <typename CorpusType>
void run_parallel_test(
    std::chrono::duration<double> duration_seq,
    const CorpusType &corpus,
    int n_gram_size,
    const std::string &strategy_name,
    int max_threads,
    ResultsExporter* exporter = nullptr);  // ← NUOVO PARAMETRO

void run_workload_scaling_test(
    const std::string& data_dir,
    int n_gram_size,
    int fixed_threads,
    const std::vector<int>& multiplier_steps,
    const std::string& strategy_name,
    ResultsExporter* exporter = nullptr);  // ← NUOVO PARAMETRO

int main(const int argc, char* argv[]) {

    if (argc < 5) {
        std::cerr << "Uso: " << argv[0] << " <directory> <n_gram_size> <max_threads> <test_mode>" << std::endl;
        std::cerr << "test_mode: SCALING o WORKLOAD" << std::endl;
        return 1;
    }

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
    Histogram hist_seq = count_seq(words, n_gram_size);
    auto end_seq = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_seq = end_seq - start_seq;

    std::cout << "\n==============================================" << std::endl;
    std::cout << "\n--- Test Sequenziale ---" << std::endl;
    std::cout << "Tempo: " << duration_seq.count() << " secondi." << std::endl;
    //print_corpus_statistics(hist_seq, n_gram_size, duration_seq.count());

    std::cout << "\n--- Test Parallelo ---" << std::endl;

    // ✅ CREA EXPORTER
    ResultsExporter exporter("results");

    if (test_mode == "SCALING"){
        std::cout << "\n==============================================" << std::endl;
        std::cout << "Esecuzione Test: Strong Scaling (Load Fixed)" << std::endl;
        std::cout << "==============================================" << std::endl;

        run_parallel_test(duration_seq, words, n_gram_size, "Static-TLS", max_threads, &exporter);
        run_parallel_test(duration_seq, doc_words, n_gram_size, "Dynamic-TLS", max_threads, &exporter);
        //run_parallel_test(duration_seq, words, n_gram_size, "Coarse-grained", max_threads, &exporter);
        run_parallel_test(duration_seq, words, n_gram_size, "Fine-grained", max_threads, &exporter);

        // ✅ SALVATAGGIO FINALE
        std::cout << "\n==============================================" << std::endl;
        std::string csv_filename = "scaling_" + std::to_string(n_gram_size) + "gram.csv";
        std::string txt_filename = "scaling_" + std::to_string(n_gram_size) + "gram_summary.txt";

        exporter.save_scaling_results(csv_filename, n_gram_size);
        exporter.save_summary(txt_filename, n_gram_size);
        std::cout << "💡 Tip: Usa 'make plot_scaling' per generare i grafici" << std::endl;

    } else if (test_mode == "WORKLOAD"){
        std::cout << "\n==============================================" << std::endl;
        std::cout << "Esecuzione Test: Workload Scaling (Threads Fixed)" << std::endl;
        std::cout << "==============================================" << std::endl;

        std::vector<int> MULTIPLIER_STEPS(10);
        int val = 1;
        for (auto& x : MULTIPLIER_STEPS) x=val, val+=1;

        run_workload_scaling_test(DATA_DIR, n_gram_size, max_threads, MULTIPLIER_STEPS, "Static-TLS", &exporter);
        run_workload_scaling_test(DATA_DIR, n_gram_size, max_threads, MULTIPLIER_STEPS, "Dynamic-TLS", &exporter);
        //run_workload_scaling_test(DATA_DIR, n_gram_size, FIXED_THREAD, MULTIPLIER_STEPS, "Coarse-grained", &exporter);
        run_workload_scaling_test(DATA_DIR, n_gram_size, max_threads, MULTIPLIER_STEPS, "Fine-grained", &exporter);

        // ✅ SALVATAGGIO FINALE
        std::cout << "\n==============================================" << std::endl;
        std::string csv_filename = "workload_" + std::to_string(n_gram_size) + "gram_t" + std::to_string(max_threads) + ".csv";
        std::string txt_filename = "workload_" + std::to_string(n_gram_size) + "gram_t" + std::to_string(max_threads) + "_summary.txt";

        exporter.save_workload_results(csv_filename, n_gram_size, max_threads);
        exporter.save_summary(txt_filename, n_gram_size);
    }

    return 0;
}

template <typename CorpusType>
void run_parallel_test(
    std::chrono::duration<double> duration_seq,
    const CorpusType& corpus,
    int n_gram_size,
    const std::string& strategy_name,
    int max_threads,
    ResultsExporter* exporter)  // ← NUOVO PARAMETRO
{
    for (int num_threads = 1; num_threads <= max_threads; ++num_threads){

        Histogram hist_par;

        auto start_par = std::chrono::high_resolution_clock::now();

        // Caso 1: Chunk-based
        if constexpr (std::is_same_v<CorpusType, std::vector<std::string>>) {
            if (strategy_name.find("Static-TLS") != std::string::npos){
                hist_par = count_par_static_tls(corpus, n_gram_size, num_threads);
            } else if (strategy_name.find("Coarse-grained") != std::string::npos){
                hist_par = count_par_coarse_grained(corpus, n_gram_size, num_threads);
            } else if (strategy_name.find("Fine-grained") != std::string::npos){
                hist_par = count_par_fine_grained(corpus, n_gram_size, num_threads);
            }
        }
        // Caso 2: Document-level
        else if constexpr (std::is_same_v<CorpusType, DocumentCorpus>) {
            hist_par = count_par_dynamic_tls(corpus, n_gram_size, num_threads);
        } else {
            std::cerr << "Errore: Tipo di corpus non riconosciuto." << std::endl;
            return;
        }

        auto end_par = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration_par = end_par - start_par;
        double speedup = duration_seq.count() / duration_par.count();
        double efficiency = speedup / num_threads;

        // ✅ STAMPA SU CONSOLE (come prima)
        std::cout << "P: " << strategy_name
                  << " | T: " << num_threads
                  << " | Tempo: " << duration_par.count() << "s"
                  << " | Speedup: " << speedup
                  << " | Efficienza: " << (efficiency * 100.0) << "%" << std::endl;

        // ✅ SALVA SU FILE (silenzioso)
        if (exporter) {
            exporter->add_result(strategy_name, num_threads,
                               duration_par.count(), speedup, efficiency);
        }
    }
}

void run_workload_scaling_test(
    const std::string &data_dir,
    int n_gram_size,
    int fixed_threads,
    const std::vector<int> &multiplier_steps,
    const std::string &strategy_name,
    ResultsExporter* exporter)  // ← NUOVO PARAMETRO
{
    for (int multiplier : multiplier_steps) {

        std::vector<std::string> chunk_words = load_and_tokenize_directory(data_dir, multiplier);
        if (chunk_words.empty()) continue;

        DocumentCorpus doc_words = load_and_tokenize_document_corpus(data_dir, multiplier);

        auto start_seq = std::chrono::high_resolution_clock::now();
        count_seq(chunk_words, n_gram_size);
        auto end_seq = std::chrono::high_resolution_clock::now();
        double t_seq = std::chrono::duration_cast<std::chrono::duration<double>>(end_seq - start_seq).count();

        auto start_par = std::chrono::high_resolution_clock::now();

        if (strategy_name.find("Static-TLS") != std::string::npos) {
            count_par_static_tls(chunk_words, n_gram_size, fixed_threads);
        } else if (strategy_name.find("Dynamic-TLS") != std::string::npos) {
            count_par_dynamic_tls(doc_words, n_gram_size, fixed_threads);
        } else if (strategy_name.find("Coarse-grained") != std::string::npos) {
            count_par_coarse_grained(chunk_words, n_gram_size, fixed_threads);
        } else if (strategy_name.find("Fine-grained") != std::string::npos) {
            count_par_fine_grained(chunk_words, n_gram_size, fixed_threads);
        }
        else{return;}

        auto end_par = std::chrono::high_resolution_clock::now();
        double t_par = std::chrono::duration_cast<std::chrono::duration<double>>(end_par - start_par).count();

        double speedup = t_seq / t_par;
        double efficiency = speedup / fixed_threads * 100.0;

        // ✅ STAMPA SU CONSOLE (come prima)
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "P: " << strategy_name
                  << " | M: " << multiplier
                  << " | T_seq: " << t_seq << "s"
                  << " | T_par: " << t_par << "s"
                  << " | Speedup: " << std::setprecision(3) << speedup
                  << " | Efficienza: " << std::setprecision(3) << efficiency << "%"
                  << std::endl;

        // ✅ SALVA SU FILE (silenzioso)
        if (exporter) {
            exporter->add_result(strategy_name, fixed_threads,
                               t_par, speedup, efficiency / 100.0, multiplier);
        }

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