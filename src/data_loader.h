#ifndef TEXT_PROCESSOR_H
#define TEXT_PROCESSOR_H

#include <string>
#include <vector>

using DocumentCorpus = std::vector<std::vector<std::string>>;

std::vector<std::string> tokenize_text(const std::string& text);

std::vector<std::string> load_and_tokenize_directory(const std::string& dirname, int multiplier = 1);

DocumentCorpus load_and_tokenize_document_corpus(const std::string& directory_path, int multiplier = 1);

#endif // TEXT_PROCESSOR_H