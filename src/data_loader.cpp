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
            
            
            std::ifstream file(entry.path());
            if (file.is_open()) {
                std::stringstream buffer;
                buffer << file.rdbuf();
                full_text_buffer += buffer.str();
                full_text_buffer += " "; 
            }
            file.close();
            
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

    
    if (!fs::exists(dir_path) || !fs::is_directory(dir_path)) {
        return {};
    }

    try {
        
        for (const auto& entry : fs::directory_iterator(dir_path)) {
            
            if (entry.is_regular_file() && entry.path().extension() == ".txt") {
                
                
                std::ifstream file(entry.path());
                if (!file.is_open()) {
                    continue;
                }
                
                std::stringstream buffer;
                buffer << file.rdbuf();
                std::string text = buffer.str();
                file.close();

                
                std::vector<std::string> words = tokenize_text(text); 

                
                if (!words.empty()) {
                    doc_corpus.push_back(words);
                    
                              
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
    
    
    #pragma omp simd
    for (size_t i = 0; i < processed_text.length(); ++i) {
        
        processed_text[i] = std::tolower(processed_text[i]);
    }

    
    #pragma omp simd
    for (size_t i = 0; i < processed_text.length(); ++i) {
        char c = processed_text[i];
        if (std::ispunct(c) || c == '\n' || c == '\t') { 
            processed_text[i] = ' ';
        }
    }
    
    
    std::stringstream ss(processed_text);
    std::string word;
    while (ss >> word) {
        if (!word.empty()) {
            words.push_back(word);
        }
    }
    
    return words;
}