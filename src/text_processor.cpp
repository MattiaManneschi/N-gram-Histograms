#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>
#include "text_processor.h"

namespace fs = std::filesystem;

std::vector<std::string> load_and_tokenize_directory(const std::string& dirname) {
    std::vector<std::string> words;
    std::string full_text_buffer;
    
    fs::path dir_path(dirname);

    if (!fs::exists(dir_path) || !fs::is_directory(dir_path)) {
        std::cerr << "Errore: Il percorso '" << dirname << "' non è una directory valida o non esiste." << std::endl;
        return {};
    }

    std::cout << "Inizio aggregazione file in: " << dirname << std::endl;


    for (const auto& entry : fs::directory_iterator(dir_path)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") {
            
            // Apertura e lettura del file
            std::ifstream file(entry.path());
            if (file.is_open()) {
                std::stringstream buffer;
                buffer << file.rdbuf();
                full_text_buffer += buffer.str();
                full_text_buffer += " "; // Aggiunge uno spazio tra i libri per separare le parole
            }
            file.close();
            //std::cout << "  - Aggregato: " << entry.path().filename().string() << std::endl;
        }
    }
    
    if (full_text_buffer.empty()) {
        std::cerr << "Attenzione: Nessun file .txt trovato nella directory " << dirname << std::endl;
        return {};
    }

    words = tokenize_text(full_text_buffer);

    std::cout << "Corpus caricato con " << words.size() << " parole." << std::endl;

    return words;
}

DocumentCorpus load_and_tokenize_document_corpus(const std::string& directory_path) {
    
    DocumentCorpus doc_corpus;
    fs::path dir_path(directory_path);

    std::cout << "Inizio caricamento del corpus a livello di documento da: " 
              << directory_path << std::endl;

    // Controllo di validità del percorso
    if (!fs::exists(dir_path) || !fs::is_directory(dir_path)) {
        std::cerr << "Errore: Il percorso '" << directory_path << "' non è una directory valida o non esiste." << std::endl;
        return {};
    }

    try {
        // Itera su tutti gli elementi nella directory
        for (const auto& entry : fs::directory_iterator(dir_path)) {
            
            if (entry.is_regular_file() && entry.path().extension() == ".txt") {
                
                // 1. Apertura e lettura del file in una singola stringa buffer
                std::ifstream file(entry.path());
                if (!file.is_open()) {
                    std::cerr << "Attenzione: Impossibile aprire il file " << entry.path().filename() << ". Saltato." << std::endl;
                    continue;
                }
                
                std::stringstream buffer;
                buffer << file.rdbuf();
                std::string text = buffer.str();
                file.close();

                // TOKENIZZAZIONE
                std::vector<std::string> words = tokenize_text(text); 

                // Aggiunge il vettore di parole del documento al Corpus (mantenendo la separazione)
                if (!words.empty()) {
                    doc_corpus.push_back(words);
                    //std::cout << "  - Caricato documento: " << entry.path().filename().string() 
                              //<< " (" << words.size() << " parole)" << std::endl;
                }
            }
        }
    } catch (const fs::filesystem_error& e) {
        std::cerr << "Errore del file system durante il caricamento: " << e.what() << std::endl;
        return {}; 
    }

    if (doc_corpus.empty()) {
        std::cerr << "Attenzione: Nessun documento valido (.txt) trovato per l'analisi." << std::endl;
    } else {
        std::cout << "Caricamento completato. Totale documenti separati nel corpus: " 
                  << doc_corpus.size() << std::endl;
    }
    
    return doc_corpus;
}

std::vector<std::string> tokenize_text(const std::string& text) {
    
    std::vector<std::string> words;
    std::string processed_text = text; 
    
    // NORMALIZZAZIONE SIMD (Lowercase)
    #pragma omp simd
    for (size_t i = 0; i < processed_text.length(); ++i) {
        // Usa std::tolower solo sui caratteri ASCII (è più veloce del locale-aware)
        processed_text[i] = std::tolower(processed_text[i]);
    }

    // NORMALIZZAZIONE SIMD (Punteggiatura e Whitespace)
    #pragma omp simd
    for (size_t i = 0; i < processed_text.length(); ++i) {
        char c = processed_text[i];
        if (std::ispunct(c) || c == '\n' || c == '\t') { 
            processed_text[i] = ' ';
        }
    }
    
    // TOKENIZZAZIONE (Sequenziale)
    std::stringstream ss(processed_text);
    std::string word;
    while (ss >> word) {
        if (!word.empty()) {
            words.push_back(word);
        }
    }
    
    return words;
}