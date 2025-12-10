#include "ngram_logic.h"
#include <omp.h>
#include <iostream>
#include <numeric>
#include <algorithm>
#include <vector>
#include <functional>

Histogram count_sequential(const std::vector<std::string>& words, int n_gram_size) {
    Histogram hist;
    if (words.size() < (size_t)n_gram_size) return hist;

    for (size_t i = 0; i <= words.size() - n_gram_size; ++i) {
        std::string n_gram = words[i];
        for (int j = 1; j < n_gram_size; ++j) {
            n_gram += " " + words[i+j];
        }
        hist[n_gram]++;
    }
    return hist;
}

Histogram count_parallel(const std::vector<std::string>& words, int n_gram_size, int requested_threads) {
    if (words.size() < (size_t)n_gram_size) return {};

    std::vector<Histogram> local_hists(requested_threads);

    // FASE DI CONTEGGIO PARALLELO
    #pragma omp parallel num_threads(requested_threads) default(none) \
        shared(words, n_gram_size, local_hists, requested_threads)
    {
        int tid = omp_get_thread_num();
        Histogram& my_hist = local_hists[tid]; 
        
        // Calcolo del limite
        size_t limit = words.size() - n_gram_size;

        // divisione del lavoro tra i thread
        #pragma omp for nowait schedule(static)
        for (size_t i = 0; i <= limit; ++i) { 
            
            // Variabile privata al loop
            std::string n_gram = words.at(i); 
            
            // Creazione della stringa N-gramma
            for (int j = 1; j < n_gram_size; ++j) {
                n_gram += " " + words.at(i+j); 
            }
            my_hist[n_gram]++; 
        }
    }

    Histogram final_hist;

    // STIMA DINAMICA DELLA DIMENSIONE TOTALE
    size_t total_unique_elements = 0;
    
    for (const auto& current_hist : local_hists) {
        total_unique_elements += current_hist.size();
    }
    
    // PRE-ALLOCAZIONE DELLA MAPPA FINALE
    // Riserva lo spazio esatto calcolato sopra. Questo evita il Re-hashing.
    final_hist.reserve(total_unique_elements); 
    
    // MERGE SEQUENZIALE
    // Il blocco 'omp master' assicura che solo il thread master esegua l'unione.
    #pragma omp master 
    {
        for (const auto& current_hist : local_hists) { 
            // Aggiorna la mappa finale
            for (const auto& pair : current_hist) {
                final_hist[pair.first] += pair.second; 
            }
        }
    }
    
    return final_hist;
}

Histogram count_parallel_document_level(const DocumentCorpus& all_document_words, int n_gram_size, int requested_threads) {
    
    if (all_document_words.empty()) {
        return {};
    }

    std::vector<Histogram> local_hists(requested_threads);
    
    // FASE DI CONTEGGIO PARALLELO (Divisione a Livello di Documento)
    #pragma omp parallel num_threads(requested_threads) default(none) \
        shared(all_document_words, n_gram_size, local_hists)
    {
        int tid = omp_get_thread_num();
        Histogram& my_hist = local_hists[tid]; 
        
        // Ogni iterazione (doc_index) è un intero documento.
        #pragma omp for schedule(dynamic)
        for (size_t doc_index = 0; doc_index < all_document_words.size(); ++doc_index) {
            
            const std::vector<std::string>& words = all_document_words[doc_index];
            
            // Se il documento è troppo piccolo
            if (words.size() < (size_t)n_gram_size) continue;
            
            size_t limit = words.size() - n_gram_size;

            // Loop SEQUENZIALE all'interno del singolo documento
            for (size_t i = 0; i <= limit; ++i) { 
                
                std::string n_gram = words.at(i); 
                
                // Creazione della stringa N-gramma
                for (int j = 1; j < n_gram_size; ++j) {
                    n_gram += " " + words.at(i+j); 
                }
                
                // Aggiornamento della mappa locale (NON richiede lock)
                my_hist[n_gram]++; 
            }
        }
    }

    
    // FASE DI MERGE SEQUENZIALE
    Histogram final_hist;

    size_t total_unique_elements = 0;
    for (const auto& current_hist : local_hists) {
        total_unique_elements += current_hist.size();
    }
    final_hist.reserve(total_unique_elements); 
    
    #pragma omp master 
    {
        for (const auto& current_hist : local_hists) { 
            for (const auto& pair : current_hist) {
                final_hist[pair.first] += pair.second; 
            }
        }
    }
    
    return final_hist;
}

Histogram count_parallel_critical_based(const std::vector<std::string>& words, int n_gram_size, int requested_threads){
    if (words.size() < (size_t)n_gram_size) {
        return {};
    }

    Histogram shared_hist;
    size_t limit = words.size() - n_gram_size;

    #pragma omp parallel num_threads(requested_threads) default(none) \
        shared(words, n_gram_size, shared_hist, limit)
    {
        #pragma omp for schedule(static) nowait
        for (size_t i = 0; i <= limit; ++i) { 
            
            std::string n_gram = words.at(i); 
            for (int j = 1; j < n_gram_size; ++j) {
                n_gram += " " + words.at(i+j); 
            }
            
            #pragma omp critical
            shared_hist[n_gram]++;
        }
    }

    return shared_hist;

}

void print_corpus_statistics(const Histogram& hist, int n_gram_size, double total_time){
    long long total_occurrences = 0;
    
    for (const auto& pair : hist) {
        total_occurrences += pair.second;
    }

    size_t unique_ngrams = hist.size();
    
    std::cout << "\n==============================================" << std::endl;
    std::cout << "Statistiche per " << n_gram_size << "-grammi" << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << "Tempo di Esecuzione Totale (Ultimo Thread): " << total_time << "s" << std::endl;
    std::cout << "Totale N-grammi (Occorrenze): " << total_occurrences << std::endl;
    std::cout << "N-grammi Unici (Vocabolario): " << unique_ngrams << std::endl;
    std::cout << "Rapporto Unici/Totali: " << (double)unique_ngrams / total_occurrences * 100.0 << "%" << std::endl;

    // 2. Statistica "Top K" (Richiede ordinamento)
    // Vettore di pair (frequenza, n-gramma)
    std::vector<std::pair<long long, std::string>> sorted_ngrams;
    for (const auto& pair : hist) {
        // {conteggio, chiave} per ordinare per conteggio
        sorted_ngrams.push_back({pair.second, pair.first}); 
    }

    // Ordina in ordine decrescente basato sul conteggio
    std::sort(sorted_ngrams.rbegin(), sorted_ngrams.rend());
    
    // Stampa i Top 10
    size_t k = 10;
    std::cout << "\nTop " << k << " " << n_gram_size << "-grammi:" << std::endl;
    for (size_t i = 0; i < k && i < sorted_ngrams.size(); ++i) {
        std::cout << i + 1 << ". '" << sorted_ngrams[i].second 
                  << "' -> " << sorted_ngrams[i].first << std::endl;
    }
    std::cout << "==============================================" << std::endl;
}