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
            std::cout << "  - Aggregato: " << entry.path().filename().string() << std::endl;
        }
    }
    
    if (full_text_buffer.empty()) {
        std::cerr << "Attenzione: Nessun file .txt trovato nella directory " << dirname << std::endl;
        return {};
    }
    
    // NORMALIZZAZIONE SIMD
    #pragma omp simd
    for (size_t i = 0; i < full_text_buffer.length(); ++i) {
        full_text_buffer[i] = std::tolower(full_text_buffer[i]);
    }

    #pragma omp simd
    for (size_t i = 0; i < full_text_buffer.length(); ++i) {
        char c = full_text_buffer[i];
        if (std::ispunct(c) || c == '\n' || c == '\t') {
            full_text_buffer[i] = ' ';
        }
    }
    
    // TOKENIZZAZIONE (Sequenziale)
    std::stringstream ss(full_text_buffer);
    std::string word;
    while (ss >> word) {
        if (!word.empty()) {
            words.push_back(word);
        }
    }
    
    return words;
}
