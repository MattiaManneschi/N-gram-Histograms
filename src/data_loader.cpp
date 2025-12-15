#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <filesystem>
#include "data_loader.h"

namespace fs = std::filesystem;

std::vector<std::string> load_and_tokenize_directory(const std::string& dirname, int multiplier) {
    std::vector<std::string> words;
    std::string full_text_buffer;
    
    fs::path dir_path(dirname);

    if (!fs::exists(dir_path) || !fs::is_directory(dir_path)) {
        return {};
    }


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

        return {};
    }

    if (multiplier > 1) {
        std::string original_text = full_text_buffer;

        size_t original_size = original_text.length();

        size_t required_size = original_size * (size_t)multiplier + (size_t)multiplier - 1;

        try {
            full_text_buffer.reserve(required_size); 
        } catch (const std::bad_alloc& e) {
            std::cerr << "ERRORE: Impossibile pre-allocare il buffer di testo per M = " << multiplier << std::endl;
            return {};
        }
        
        for (int i = 1; i < multiplier; ++i) { 
            full_text_buffer += " " + original_text; 
        }
    }

    words = tokenize_text(full_text_buffer);

    return words;
}

DocumentCorpus load_and_tokenize_document_corpus(const std::string& directory_path, int multiplier) {
    
    DocumentCorpus doc_corpus;

    fs::path dir_path(directory_path);

    // Controllo di validità del percorso
    if (!fs::exists(dir_path) || !fs::is_directory(dir_path)) {
        return {};
    }

    try {
        // Itera su tutti gli elementi nella directory
        for (const auto& entry : fs::directory_iterator(dir_path)) {
            
            if (entry.is_regular_file() && entry.path().extension() == ".txt") {
                
                // 1. Apertura e lettura del file in una singola stringa buffer
                std::ifstream file(entry.path());
                if (!file.is_open()) {
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
        return {}; 
    }

    if (doc_corpus.empty()) {
        return {};
    }
    else
    {
        if (multiplier > 1) {

            const DocumentCorpus original_base = doc_corpus;
            doc_corpus.clear();

            for (int i = 0; i < multiplier; ++i) {
                DocumentCorpus corpus_base = doc_corpus;
                doc_corpus.insert(doc_corpus.end(), original_base.begin(), original_base.end());
            }
        }
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