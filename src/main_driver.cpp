#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include "text_processor.h"
#include "ngram_logic.h"
#include "results_exports.h"
#include <type_traits>
#include <iomanip>

//TODO AGGIUNGERE DISTRIBUZIONI E STATISTICHE UTILI (VEDERE PDF O VECCHI PROGETTI)
//TODO OTTIMIZZAZIONE CODICE (NOMI FILE, NOMI FUNZIONI, ETC...)

template <typename CorpusType>
void run_thread_scaling_test(
    const CorpusType& corpus,
    int n_gram_size,
    const std::string& strategy_name,
    int max_threads,
    std::chrono::duration<double> duration_seq,
    ResultsExporter* exporter);

void run_workload_scaling_test(
    const std::string& data_dir,
    int n_gram_size,
    int fixed_threads,
    const std::vector<int>& multiplier_steps,
    const std::string& strategy_name,
    const std::vector<double>& sequential_times,
    ResultsExporter* exporter = nullptr);

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

    std::cout << "\n===========================================" << std::endl;
    std::cout << "Analisi " << n_gram_size << "-grammi" << std::endl;
    std::cout << "\nDati caricati da: " << DATA_DIR << std::endl;
    std::cout << "===========================================" << std::endl;

    ResultsExporter exporter("results");

    if (test_mode == "THREAD"){
        std::cout << "\n==============================================" << std::endl;
        std::cout << "Esecuzione Test: Strong Scaling (Load Fixed)" << std::endl;
        std::cout << "==============================================" << std::endl;

        auto words = load_and_tokenize_directory(DATA_DIR);
        DocumentCorpus doc_words = load_and_tokenize_document_corpus(DATA_DIR);

        if (words.empty())
        {
            std::cerr << "Errore: nessun dato caricato" << std::endl;
            return 1;
        }

        std::cout << "--- Test Sequenziale (Baseline) ---" << std::endl;
        const auto start_seq = std::chrono::high_resolution_clock::now();
        Histogram hist_seq = count_seq(words, n_gram_size);
        const auto end_seq = std::chrono::high_resolution_clock::now();
        const double sequential_time = std::chrono::duration<double>(end_seq - start_seq).count();

        std::cout << "Tempo sequenziale: " << sequential_time << " secondi\n" << std::endl;

        std::cout << "--- Test Paralleli --- \n" << std::endl;

        run_thread_scaling_test(words, n_gram_size, "Static-TLS", max_threads, std::chrono::duration<double>(sequential_time), &exporter);
        run_thread_scaling_test(doc_words, n_gram_size, "Dynamic-TLS", max_threads, std::chrono::duration<double>(sequential_time), &exporter);
        //run_parallel_test(duration_seq, words, n_gram_size, "Coarse-grained", max_threads, &exporter);
        run_thread_scaling_test(words, n_gram_size, "Fine-grained", max_threads, std::chrono::duration<double>(sequential_time), &exporter);

        // ✅ SALVATAGGIO FINALE
        std::cout << "\n==============================================" << std::endl;
        std::string csv_filename = "thread_scaling_" + std::to_string(n_gram_size) + "gram.csv";
        std::string txt_filename = "thread_scaling_" + std::to_string(n_gram_size) + "gram_summary.txt";

        exporter.save_scaling_results(csv_filename, n_gram_size);
        exporter.save_summary(txt_filename, n_gram_size);

    } else if (test_mode == "WORKLOAD"){
        std::cout << "\n==============================================" << std::endl;
        std::cout << "Esecuzione Test: Workload Scaling (Threads Fixed)" << std::endl;
        std::cout << "==============================================" << std::endl;

        std::vector<int> MULTIPLIER_STEPS(10);
        int val = 1;
        for (auto& x : MULTIPLIER_STEPS) x=val, val+=1;

        std::cout << "\n--- Test sequenziali ---\n" << std::endl;

        std::vector<double> sequential_times;

        sequential_times.reserve(MULTIPLIER_STEPS.size());

        for (int multiplier : MULTIPLIER_STEPS)
        {
            std::vector<std::string> words = load_and_tokenize_directory(DATA_DIR, multiplier);

            if (words.empty())
            {
                std::cerr << "Errore: caricamento fallito per multiplier=" << multiplier << std::endl;
                sequential_times.push_back(0.0); // Placeholder
                continue;
            }

            auto start_seq = std::chrono::high_resolution_clock::now();
            count_seq(words, n_gram_size);
            auto end_seq = std::chrono::high_resolution_clock::now();
            double seq_time = std::chrono::duration<double>(end_seq - start_seq).count();

            sequential_times.push_back(seq_time);

            std::cout << "M =" << std::setw(2) << multiplier
                      << " | Tempo sequenziale: " << std::fixed << std::setprecision(4)
                      << seq_time << "s" << std::endl;

            words.clear();
            words.shrink_to_fit();
        }

        std::cout << "\n--- Test Paralleli ---\n" << std::endl;

        run_workload_scaling_test(DATA_DIR, n_gram_size, max_threads, MULTIPLIER_STEPS, "Static-TLS", sequential_times, &exporter);
        run_workload_scaling_test(DATA_DIR, n_gram_size, max_threads, MULTIPLIER_STEPS, "Dynamic-TLS", sequential_times, &exporter);
        //run_workload_scaling_test(DATA_DIR, n_gram_size, FIXED_THREAD, MULTIPLIER_STEPS, "Coarse-grained", &exporter);
        run_workload_scaling_test(DATA_DIR, n_gram_size, max_threads, MULTIPLIER_STEPS, "Fine-grained", sequential_times, &exporter);

        // ✅ SALVATAGGIO FINALE
        std::cout << "\n==============================================" << std::endl;
        std::string csv_filename = "workload_" + std::to_string(n_gram_size) + "gram_t" + std::to_string(max_threads) + ".csv";
        std::string txt_filename = "workload_" + std::to_string(n_gram_size) + "gram_t" + std::to_string(max_threads) + "_summary.txt";

        exporter.save_workload_results(csv_filename, n_gram_size, max_threads);
        exporter.save_summary(txt_filename, n_gram_size);
    } else
    {
        std::cerr << "Errore: test mode non riconosciuto. Usa THREAD o WORKLOAD ---\n" << std::endl;
    }

    return 0;
}

template <typename CorpusType>
void run_thread_scaling_test(
    const CorpusType& corpus,
    int n_gram_size,
    const std::string& strategy_name,
    const int max_threads,
    const std::chrono::duration<double> duration_seq,
    ResultsExporter* exporter)
{
    std::cout << "Strategia: " << strategy_name << std::endl;

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
        std::cout << "T: " << num_threads
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
    const std::string& data_dir,
    int n_gram_size,
    int fixed_threads,
    const std::vector<int>& multiplier_steps,
    const std::string& strategy_name,
    const std::vector<double>& sequential_times,
    ResultsExporter* exporter)
{
    std::cout << "Strategia: " << strategy_name << " (Threads=" << fixed_threads << ")" << std::endl;

    for (size_t i = 0; i < multiplier_steps.size(); ++i) {
        int multiplier = multiplier_steps[i];
        double sequential_time = sequential_times[i];

        // CARICA DATI CON IL MOLTIPLICATORE CORRENTE
        std::vector<std::string> chunk_words = load_and_tokenize_directory(data_dir, multiplier);
        if (chunk_words.empty()) {
            std::cerr << "  Errore: caricamento fallito per multiplier=" << multiplier << std::endl;
            continue;
        }

        DocumentCorpus doc_words = load_and_tokenize_document_corpus(data_dir, multiplier);

        // ESEGUI SOLO TEST PARALLELO (il sequenziale è già stato fatto)
        auto start_par = std::chrono::high_resolution_clock::now();

        if (strategy_name.find("Static-TLS") != std::string::npos) {
            count_par_static_tls(chunk_words, n_gram_size, fixed_threads);
        } else if (strategy_name.find("Dynamic-TLS") != std::string::npos) {
            count_par_dynamic_tls(doc_words, n_gram_size, fixed_threads);
        } else if (strategy_name.find("Coarse-Grained") != std::string::npos) {
            count_par_coarse_grained(chunk_words, n_gram_size, fixed_threads);
        } else if (strategy_name.find("Fine-grained") != std::string::npos) {
            count_par_fine_grained(chunk_words, n_gram_size, fixed_threads);
        } else {
            std::cerr << "  Errore: Strategia non riconosciuta" << std::endl;
            return;
        }

        auto end_par = std::chrono::high_resolution_clock::now();
        double parallel_time = std::chrono::duration<double>(end_par - start_par).count();

        // Calcola metriche
        double speedup = sequential_time / parallel_time;
        double efficiency = (speedup / fixed_threads) * 100.0;

        // Stampa risultati
        std::cout << "M =" << std::setw(2) << multiplier
                  << " | T_par: " << parallel_time << "s"
                  << " | Speedup: " << std::setprecision(2) << speedup
                  << " | Efficienza: " << std::setprecision(1) << efficiency << "%"
                  << std::endl;

        // Salva su file
        if (exporter) {
            exporter->add_result(strategy_name, fixed_threads,
                               parallel_time, speedup, efficiency / 100.0, multiplier);
        }

        // CLEANUP MEMORIA
        chunk_words.clear();
        chunk_words.shrink_to_fit();

        doc_words.clear();
        doc_words.shrink_to_fit();
    }

    std::cout << std::endl; // Separatore tra strategie
}