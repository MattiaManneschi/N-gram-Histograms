#ifndef TEXT_PROCESSOR_H
#define TEXT_PROCESSOR_H

#include <string>
#include <vector>

/**
 * Carica TUTTI i file .txt da una directory specificata, esegue la tokenization 
 * e la normalizzazione su tutto il corpus aggregato.
 * @param dirname Il percorso della directory contenente i file di testo (es. "data/").
 * @return Un vettore di stringhe (parole) pronte per l'analisi.
 */
std::vector<std::string> load_and_tokenize_directory(const std::string& dirname); // Nuova firma

#endif // TEXT_PROCESSOR_H