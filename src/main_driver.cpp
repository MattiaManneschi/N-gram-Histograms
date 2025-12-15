#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include "data_loader.h"
#include "ngram_counter.h"
#include "results_exports.h"
#include <iomanip>

//TODO OTTIMIZZARE SPEEDUP THREAD SCALING (UN PO TUTTE) E WORKLOAD SCALING (VERSIONE CHUNK-BASED)
//TODO AGGIUNGERE DISTRIBUZIONI E STATISTICHE UTILI (VEDERE PDF O VECCHI PROGETTI)

const std::string DATA_DIR = "data/Texts";


void run_thread_scaling_test(
    int n_gram_size,
    const std::string& strategy_name,
    int max_threads,
    double sequential_time,
    ResultsExporter* exporter);

void run_workload_scaling_test(
    int n_gram_size,
    int fixed_threads,
    const std::vector<int>& multiplier_steps,
    const std::string& strategy_name,
    const std::vector<double>& sequential_times,
    ResultsExporter* exporter = nullptr);

void export_data (int n_gram_size, int max_threads, const ResultsExporter* exporter);

std::vector<double> get_sequential_times_per_multiplier(const std::vector<int>& MULTIPLIER_STEPS, int n_gram_size);

int main(const int argc, char* argv[]) {

    if (argc < 5) {
        std::cerr << "Uso: " << argv[0] << " <directory> <n_gram_size> <max_threads> <test_mode>" << std::endl;
        std::cerr << "test_mode: SCALING o WORKLOAD" << std::endl;
        return 1;
    }

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
        std::cout << "Esecuzione Test: Thread Scaling (Load Fixed)" << std::endl;
        std::cout << "==============================================" << std::endl;

        std::cout << "\n--- Test Sequenziale ---" << std::endl;

        std::vector<double> sequential_times;
        const auto words = load_and_tokenize_directory(DATA_DIR);

        /*for (int i = 0; i < 10; i++)
        {
            auto start_seq = std::chrono::high_resolution_clock::now();
            count_seq(words, n_gram_size);
            auto end_seq = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration_seq = end_seq - start_seq;

            std::cout << "I = " << i << " | T = " << duration_seq.count() << std::endl;

            sequential_times.push_back(duration_seq.count());
        }*/

        //const double sequential_time = std::accumulate(sequential_times.begin(), sequential_times.end(), 0.0) / static_cast<double>(10);

        constexpr double sequential_time = 6.25;

        std::cout << "\nTempo sequenziale medio: " << sequential_time << " secondi\n" << std::endl;

        std::cout << "--- Test Parallelo --- \n" << std::endl;

        run_thread_scaling_test(n_gram_size, "Chunk-based-TLS", max_threads, sequential_time, &exporter);
        run_thread_scaling_test(n_gram_size, "Document-level-TLS", max_threads, sequential_time, &exporter);
        run_thread_scaling_test(n_gram_size, "Fine-grained-locking", max_threads, sequential_time, &exporter);

        export_data(n_gram_size, max_threads, &exporter);

    } else if (test_mode == "WORKLOAD"){
        std::cout << "\n==============================================" << std::endl;
        std::cout << "Esecuzione Test: Workload Scaling (Threads Fixed)" << std::endl;
        std::cout << "==============================================" << std::endl;

        std::vector<int> MULTIPLIER_STEPS(10);
        int val = 1;
        for (auto& x : MULTIPLIER_STEPS) x=val, val+=1;

        std::cout << "\n--- Test sequenziale ---\n" << std::endl;

        //const std::vector<double> sequential_times = get_sequential_times_per_multiplier(MULTIPLIER_STEPS, n_gram_size);

        const std::vector sequential_times = {5.93, 10.34, 15.03, 19.81, 24.27, 30.81, 36.84, 40.00, 45.98, 49.52};

        std::cout << "\n--- Test Parallelo ---\n" << std::endl;

        //run_workload_scaling_test(n_gram_size, max_threads, MULTIPLIER_STEPS, "Chunk-based-TLS", sequential_times, &exporter);
        //run_workload_scaling_test(n_gram_size, max_threads, MULTIPLIER_STEPS, "Document-level-TLS", sequential_times, &exporter);
        run_workload_scaling_test(n_gram_size, max_threads, MULTIPLIER_STEPS, "Fine-grained-locking", sequential_times, &exporter);

        export_data(n_gram_size, max_threads, &exporter);
    } else
    {
        std::cerr << "Errore: test mode non riconosciuto. Usa THREAD o WORKLOAD ---\n" << std::endl;
    }

    return 0;
}

void run_thread_scaling_test(
    const int n_gram_size,
    const std::string& strategy_name,
    const int max_threads,
    const double sequential_time,
    ResultsExporter* exporter)
{
    std::cout << "Strategia: " << strategy_name << std::endl;
    for (int num_threads = 1; num_threads <= max_threads; ++num_threads){
        auto start_par = std::chrono::high_resolution_clock::now();

        if (strategy_name.find("Chunk-based-TLS") != std::string::npos) {
            count_par_chunk_based_tls(DATA_DIR, n_gram_size, num_threads, 1);
        } else if (strategy_name.find("Document-level-TLS") != std::string::npos) {
            count_par_document_level_tls(DATA_DIR, n_gram_size, num_threads, 1);
        } else if (strategy_name.find("Fine-grained-locking") != std::string::npos) {
            count_par_fine_grained_locking(DATA_DIR, n_gram_size, num_threads, 1);
        } else {
            std::cerr << "  Errore: Strategia non riconosciuta" << std::endl;
            return;
        }

        auto end_par = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration_par = end_par - start_par;
        const double speedup = sequential_time / duration_par.count();
        const double efficiency = speedup / num_threads;

        std::cout << "T: " << num_threads
                  << " | Tempo: " << duration_par.count() << "s"
                  << " | Speedup: " << speedup << std::endl;

        if (exporter) {
            exporter->add_result(strategy_name, num_threads,
                               duration_par.count(), speedup, efficiency);
        }
    }
}

void run_workload_scaling_test(
    const int n_gram_size,
    const int fixed_threads,
    const std::vector<int>& multiplier_steps,
    const std::string& strategy_name,
    const std::vector<double>& sequential_times,
    ResultsExporter* exporter)
{
    std::cout << "Strategia: " << strategy_name << " (Threads=" << fixed_threads << ")" << std::endl;

    for (size_t i = 0; i < multiplier_steps.size(); ++i) {
        const int multiplier = multiplier_steps[i];
        const double sequential_time = sequential_times[i];

        auto start_par = std::chrono::high_resolution_clock::now();

        if (strategy_name.find("Chunk-based-TLS") != std::string::npos) {
            count_par_chunk_based_tls(DATA_DIR, n_gram_size, fixed_threads, multiplier);
        } else if (strategy_name.find("Document-level-TLS") != std::string::npos) {
            count_par_document_level_tls(DATA_DIR, n_gram_size, fixed_threads, multiplier);
        } else if (strategy_name.find("Fine-grained-locking") != std::string::npos) {
            count_par_fine_grained_locking(DATA_DIR, n_gram_size, fixed_threads, multiplier);
        } else {
            std::cerr << "  Errore: Strategia non riconosciuta" << std::endl;
            return;
        }

        auto end_par = std::chrono::high_resolution_clock::now();
        const double parallel_time = std::chrono::duration<double>(end_par - start_par).count();

        const double speedup = sequential_time / parallel_time;
        const double efficiency = (speedup / fixed_threads) * 100.0;

        std::cout << "M =" << std::setw(2) << multiplier
                  << " | T_par: " << std::fixed << std::setprecision(3) << parallel_time << "s"
                  << " | Speedup: " << std::setprecision(2) << speedup << std::endl;

        if (exporter) {
            exporter->add_result(strategy_name, fixed_threads,
                               parallel_time, speedup, efficiency / 100.0, multiplier);
        }
    }

    std::cout << std::endl;
}

std::vector<double> get_sequential_times_per_multiplier(const std::vector<int>& MULTIPLIER_STEPS, const int n_gram_size)
{
    std::vector<double> sequential_times;

    sequential_times.reserve(MULTIPLIER_STEPS.size());

    for (const int multiplier : MULTIPLIER_STEPS)
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

        std::cout <<"M =" << std::setw(2) << multiplier
                  << " | T_seq: " << seq_time << "s" << std::endl;

        words.clear();
        words.shrink_to_fit();
    }

    return sequential_times;
}


void export_data(const int n_gram_size, const int max_threads, const ResultsExporter* exporter)
{
    std::cout << "\n==============================================" << std::endl;
    const std::string csv_filename = "workload_" + std::to_string(n_gram_size) + "gram_t" + std::to_string(max_threads) + ".csv";
    const std::string txt_filename = "workload_" + std::to_string(n_gram_size) + "gram_t" + std::to_string(max_threads) + "_summary.txt";

    exporter->save_workload_results(csv_filename, n_gram_size, max_threads);
    exporter->save_summary(txt_filename, n_gram_size);
}
