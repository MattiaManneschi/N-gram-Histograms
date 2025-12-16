#include <iostream>
#include <fstream>
#include <chrono>
#include <algorithm>
#include "data_loader.h"
#include "ngram_counter.h"
#include "results_exports.h"
#include <iomanip>
#include <numeric>
#include <omp.h>

//TODO AGGIUNGERE DISTRIBUZIONI E STATISTICHE UTILI (VEDERE PDF O VECCHI PROGETTI)
//TODO OTTIMIZZARE WORKLOAD SCALING (DOCUMENT-LEVEL)

const std::string DATA_DIR = "data/Texts";
constexpr int TEST_ITER = 1;


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

std::vector<double> get_sequential_times_per_multiplier(const std::vector<int>& MULTIPLIER_STEPS, int n_gram_size);

int main(const int argc, char* argv[]) {

    const int n_gram_size = std::stoi(argv[2]);
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

        /*for (int i = 0; i < TEST_ITER; i++)
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

        run_thread_scaling_test(n_gram_size, "Hybrid-TLS" , max_threads, sequential_time, &exporter);
        run_thread_scaling_test(n_gram_size, "Single Reader" , max_threads, sequential_time, &exporter);
        run_thread_scaling_test(n_gram_size, "On the Fly" , max_threads, sequential_time, &exporter);

        std::cout << "\n==============================================" << std::endl;
        const std::string csv_filename = "thread_" + std::to_string(n_gram_size) + "gram_t" + std::to_string(max_threads) + ".csv";
        const std::string txt_filename = "thread_" + std::to_string(n_gram_size) + "gram_t" + std::to_string(max_threads) + "_summary.txt";

        exporter.save_scaling_results(csv_filename, n_gram_size);
        exporter.save_summary(txt_filename, n_gram_size);

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

        run_workload_scaling_test(n_gram_size, max_threads, MULTIPLIER_STEPS, "Hybrid-TLS", sequential_times, &exporter);
        run_workload_scaling_test(n_gram_size, max_threads, MULTIPLIER_STEPS, "Document-level-TLS", sequential_times, &exporter);
        run_workload_scaling_test(n_gram_size, max_threads, MULTIPLIER_STEPS, "Fine-grained-locking", sequential_times, &exporter);

        std::cout << "\n==============================================" << std::endl;
        const std::string csv_filename = "workload_" + std::to_string(n_gram_size) + "gram_t" + std::to_string(max_threads) + ".csv";
        const std::string txt_filename = "workload_" + std::to_string(n_gram_size) + "gram_t" + std::to_string(max_threads) + "_summary.txt";

        exporter.save_workload_results(csv_filename, n_gram_size, max_threads);
        exporter.save_summary(txt_filename, n_gram_size);
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
    std::cout << "Strategia: " << strategy_name << " (Threads=" << max_threads << ")" << std::endl;

    Histogram hist;

    for (int num_threads = 1; num_threads <= max_threads; ++num_threads){
        omp_set_num_threads(num_threads);
        std::vector<double> par_times;

        for ( int i = 0; i < TEST_ITER; i++)
        {
            const auto start_par = omp_get_wtime();
            if (strategy_name.find("Hybrid-TLS") != std::string::npos) {
                count_par_hybrid_preload_TLS(hist, n_gram_size);
            } else if (strategy_name.find("Single Reader") != std::string::npos) {
                count_par_singleReader_Worker_TLS(hist, n_gram_size);
            } else if (strategy_name.find("On the Fly") != std::string::npos) {
                count_par_onTheFly_parallelIO(hist, n_gram_size);
            }
            const auto end_par = omp_get_wtime();
            const auto duration_par = end_par - start_par;
            par_times.push_back(duration_par);
        }

        const double duration_par = std::accumulate(par_times.begin(), par_times.end(), 0.0) / par_times.size();

        const double speedup = sequential_time / duration_par;
        const double efficiency = speedup / num_threads;

        std::cout << "TH: " << num_threads
                  << " | N.TEST: " << TEST_ITER
                  << " | Tempo medio: " << duration_par << "s"
                  << " | Speedup medio: " << speedup << std::endl;

        if (exporter) {
            exporter->add_result(strategy_name, num_threads,
                               duration_par, speedup, efficiency);
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

        const auto start_par = omp_get_wtime();

        if (strategy_name.find("Hybrid-TLS") != std::string::npos) {
            Histogram hist;
            count_par_singleReader_Worker_TLS(hist,n_gram_size, multiplier);
        } else if (strategy_name.find("Document-level-TLS") != std::string::npos) {
            count_par_document_level_tls(DATA_DIR, n_gram_size, fixed_threads, multiplier);
        } else if (strategy_name.find("Fine-grained-locking") != std::string::npos) {
            count_par_fine_grained_locking(DATA_DIR, n_gram_size, fixed_threads, multiplier);
        }

        const auto end_par = omp_get_wtime();
        const auto duration_par = end_par - start_par;

        const double speedup = sequential_time / duration_par;
        const double efficiency = (speedup / fixed_threads) * 100.0;

        std::cout << "M =" << std::setw(2) << multiplier
                  << " | T_par: " << std::fixed << std::setprecision(3) << duration_par << "s"
                  << " | Speedup: " << std::setprecision(2) << speedup << std::endl;

        if (exporter) {
            exporter->add_result(strategy_name, fixed_threads,
                               duration_par, speedup, efficiency / 100.0, multiplier);
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
