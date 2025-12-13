#include "ngram_logic.h"
#include <omp.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <functional>

//Sequenziale
Histogram count_seq(const std::vector<std::string>& words, const int n_gram_size) {
    Histogram hist;
    if (words.size() < static_cast<size_t>(n_gram_size)) return hist;

    for (size_t i = 0; i <= words.size() - n_gram_size; ++i) {
        std::string n_gram = words[i];
        for (int j = 1; j < n_gram_size; ++j) {
            n_gram += " " + words[i+j];
        }
        hist[n_gram]++;
    }
    return hist;
}

//Parallele - TLS based
Histogram count_par_static_tls(const std::vector<std::string>& words, int n_gram_size, int num_threads) {
    if (words.size() < static_cast<size_t>(n_gram_size)) return {};

    std::vector<Histogram> local_hists(num_threads);

    // FASE DI CONTEGGIO PARALLELO
    #pragma omp parallel num_threads(num_threads) default(none) \
        shared(words, n_gram_size, local_hists, num_threads)
    {
        const int tid = omp_get_thread_num();
        Histogram& my_hist = local_hists[tid]; 
        
        // Calcolo del limite
        const size_t limit = words.size() - n_gram_size;

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
            for (const auto& [fst, snd] : current_hist) {
                final_hist[fst] += snd;
            }
        }
    }
    
    return final_hist;
}

Histogram count_par_dynamic_tls(const DocumentCorpus& doc_words, int ngram_size, int num_threads) {
    
    if (doc_words.empty()) {
        return {};
    }

    std::vector<Histogram> local_hists(num_threads);
    
    // FASE DI CONTEGGIO PARALLELO (Divisione a Livello di Documento)
    #pragma omp parallel num_threads(num_threads) default(none) \
        shared(doc_words, ngram_size, local_hists)
    {
        int tid = omp_get_thread_num();
        Histogram& my_hist = local_hists[tid]; 
        
        // Ogni iterazione (doc_index) è un intero documento.
        #pragma omp for schedule(dynamic)
        for (size_t doc_index = 0; doc_index < doc_words.size(); ++doc_index) {
            
            const std::vector<std::string>& words = doc_words[doc_index];
            
            // Se il documento è troppo piccolo
            if (words.size() < static_cast<size_t>(ngram_size)) continue;

            const size_t limit = words.size() - ngram_size;

            // Loop SEQUENZIALE all'interno del singolo documento
            for (size_t i = 0; i <= limit; ++i) { 
                
                std::string n_gram = words.at(i); 
                
                // Creazione della stringa N-gramma
                for (int j = 1; j < ngram_size; ++j) {
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
            for (const auto& [fst, snd] : current_hist) {
                final_hist[fst] += snd;
            }
        }
    }
    
    return final_hist;
}

//Parallele - Lock based
Histogram count_par_coarse_grained(const std::vector<std::string>& words, int n_gram_size, int requested_threads){
    if (words.size() < static_cast<size_t>(n_gram_size)) {
        return {};
    }

    Histogram shared_hist;
    size_t limit = words.size() - n_gram_size;

    #pragma omp parallel num_threads(requested_threads) default(none) \
        shared(words, n_gram_size, shared_hist, limit)
    {
        // Ogni thread ha il suo buffer locale
        Histogram local_buffer;

        #pragma omp for schedule(static) nowait
        for (size_t i = 0; i <= limit; ++i) {

            std::string n_gram = words.at(i);
            for (int j = 1; j < n_gram_size; ++j) {
                n_gram += " " + words.at(i+j);
            }

            // Accumula localmente
            local_buffer[n_gram]++;

            // Flush periodico al shared_hist
            if (constexpr size_t BATCH_SIZE = 1000; local_buffer.size() >= BATCH_SIZE) {
                #pragma omp critical
                {
                    for (const auto& [fst, snd] : local_buffer) {
                        shared_hist[fst] += snd;
                    }
                }
                local_buffer.clear();
            }
        }

        // Flush finale
        if (!local_buffer.empty()) {
            #pragma omp critical
            {
                for (const auto& [fst, snd] : local_buffer) {
                    shared_hist[fst] += snd;
                }
            }
        }
    }

    return shared_hist;
}

Histogram count_par_fine_grained(const std::vector<std::string>& words, int n_gram_size, int num_threads)
{
    if (words.size() < static_cast<size_t>(n_gram_size)) {
        return {};
    }

    Histogram shared_hist;
    size_t limit = words.size() - n_gram_size;

    // Array di lock - NUMERO PRIMO per migliore distribuzione hash
    // Perché primo? Riduce collisioni con pattern periodici nei dati
    constexpr int NUM_LOCKS = 1021;
    std::vector<omp_lock_t> locks(NUM_LOCKS);

    // Inizializza tutti i lock
    for (int i = 0; i < NUM_LOCKS; ++i) {
        omp_init_lock(&locks[i]);
    }

    #pragma omp parallel num_threads(num_threads) default(none) \
        shared(words, n_gram_size, shared_hist, limit, locks)
    {
        // BATCH PROCESSING LOCALE
        // Ogni thread accumula n-grammi nel suo buffer privato
        // per ridurre il numero di accessi alla mappa condivisa
        std::unordered_map<std::string, long long> local_buffer;

#pragma omp for schedule(static) nowait
        for (size_t i = 0; i <= limit; ++i) {

            // Costruzione n-gramma (lavoro locale, no sync)
            std::string n_gram = words.at(i);
            for (int j = 1; j < n_gram_size; ++j) {
                n_gram += " " + words.at(i+j);
            }

            // Accumula localmente (no lock necessario)
            local_buffer[n_gram]++;

            // FLUSH PERIODICO quando il buffer locale è pieno
            if (constexpr size_t BATCH_SIZE = 10000; local_buffer.size() >= BATCH_SIZE) {

                // Per ogni n-gramma nel buffer locale
                for (const auto& [fst, snd] : local_buffer) {

                    // LOCK STRIPING: Hash → Lock ID
                    // Usa std::hash per ottenere un valore hash
                    const size_t hash_val = std::hash<std::string>{}(fst);

                    // Mappa hash a un lock specifico (modulo NUM_LOCKS)
                    const int lock_id = hash_val % NUM_LOCKS;

                    // CRITICAL REGION (ma solo per questo lock!)
                    // Solo thread che vogliono aggiornare n-grammi con lo stesso
                    // lock_id si bloccano a vicenda
                    omp_set_lock(&locks[lock_id]);
                    shared_hist[fst] += snd;
                    omp_unset_lock(&locks[lock_id]);
                }

                // Svuota il buffer locale
                local_buffer.clear();
            }
        }

        // FLUSH FINALE: processa gli n-grammi rimanenti nel buffer
        for (const auto& [fst, snd] : local_buffer) {
            const size_t hash_val = std::hash<std::string>{}(fst);
            const int lock_id = hash_val % NUM_LOCKS;

            omp_set_lock(&locks[lock_id]);
            shared_hist[fst] += snd;
            omp_unset_lock(&locks[lock_id]);
        }
    }

    // CLEANUP: Distruggi tutti i lock
    for (int i = 0; i < NUM_LOCKS; ++i) {
        omp_destroy_lock(&locks[i]);
    }

    return shared_hist;
}

//stampa statistiche
void print_corpus_statistics(const Histogram& hist, const int n_gram_size, const double total_time){
    long long total_occurrences = 0;
    
    for (const auto& [fst, snd] : hist) {
        total_occurrences += snd;
    }

    const size_t unique_ngrams = hist.size();
    
    std::cout << "\n==============================================" << std::endl;
    std::cout << "Statistiche per " << n_gram_size << "-grammi" << std::endl;
    std::cout << "==============================================" << std::endl;
    std::cout << "Tempo di Esecuzione Totale (Ultimo Thread): " << total_time << "s" << std::endl;
    std::cout << "Totale N-grammi (Occorrenze): " << total_occurrences << std::endl;
    std::cout << "N-grammi Unici (Vocabolario): " << unique_ngrams << std::endl;
    std::cout << "Rapporto Unici/Totali: " << static_cast<double>(unique_ngrams) / total_occurrences * 100.0 << "%" << std::endl;

    // 2. Statistica "Top K" (Richiede ordinamento)
    // Vettore di pair (frequenza, n-gramma)
    std::vector<std::pair<long long, std::string>> sorted_ngrams;
    for (const auto& [fst, snd] : hist) {
        // {conteggio, chiave} per ordinare per conteggio
        sorted_ngrams.push_back({snd, fst});
    }

    // Ordina in ordine decrescente basato sul conteggio
    std::sort(sorted_ngrams.rbegin(), sorted_ngrams.rend());
    
    // Stampa i Top 10
    const size_t k = 10;
    std::cout << "\nTop " << k << " " << n_gram_size << "-grammi:" << std::endl;
    for (size_t i = 0; i < k && i < sorted_ngrams.size(); ++i) {
        std::cout << i + 1 << ". '" << sorted_ngrams[i].second 
                  << "' -> " << sorted_ngrams[i].first << std::endl;
    }
    std::cout << "==============================================" << std::endl;
}