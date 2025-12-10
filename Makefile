# Makefile per Progetto Mid-Term (Bigrammi/Trigrammi)

CXX = g++
CXXFLAGS = -std=c++17 -Wall -O3
OPENMP_FLAG = -fopenmp

TARGET = ngram_analyzer
SRC_DIR = src
BIN_DIR = bin
SRCS = $(wildcard $(SRC_DIR)/*.cpp)
NTHREADS ?= 16
NGRAM_SIZE ?= 2
DATA_DIR ?= data/Texts

.PHONY: all seq par clean run run_seq

# Regola per la compilazione della versione parallela (con OpenMP)
$(BIN_DIR)/$(TARGET)_par: $(SRCS) | $(BIN_DIR)
	@echo "Compiling PARALLEL AND SEQUENTIAL versions..."
	$(CXX) $(CXXFLAGS) $(OPENMP_FLAG) $^ -o $@ 

# Regola per la compilazione della versione sequenziale (senza OpenMP)
$(BIN_DIR)/$(TARGET)_seq: $(SRCS) | $(BIN_DIR)
	@echo "Compiling SEQUENTIAL version..."
	$(CXX) $(CXXFLAGS) $^ -o $@

# Creazione della directory bin
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Esecuzione della versione parallela
run: $(BIN_DIR)/$(TARGET)_par
	@echo "Esecuzione Parallela: NGRAM_SIZE=$(NGRAM_SIZE), NTHREADS=$(NTHREADS), MODE=$(TEST_MODE)"
	./$(BIN_DIR)/$(TARGET)_par $(DATA_DIR) $(NGRAM_SIZE) $(NTHREADS) $(TEST_MODE)

clean:
	rm -rf $(BIN_DIR)