# Makefile per Progetto Mid-Term (Bigrammi/Trigrammi)

CXX = g++
CXXFLAGS = -std=c++17 -Wall -O3
OPENMP_FLAG = -fopenmp

TARGET = ngram_analyzer
SRC_DIR = src
BIN_DIR = bin
SRCS = $(wildcard $(SRC_DIR)/*.cpp)

.PHONY: all seq par clean run run_seq

# Regola per la compilazione della versione parallela (con OpenMP)
$(BIN_DIR)/$(TARGET)_par: $(SRCS) | $(BIN_DIR)
	@echo "Compiling PARALLEL version..."
	$(CXX) $(CXXFLAGS) $(OPENMP_FLAG) $^ -o $@ 

# Regola per la compilazione della versione sequenziale (senza OpenMP)
$(BIN_DIR)/$(TARGET)_seq: $(SRCS) | $(BIN_DIR)
	@echo "Compiling SEQUENTIAL version..."
	$(CXX) $(CXXFLAGS) $^ -o $@

# Creazione della directory bin
$(BIN_DIR):
	mkdir -p $(BIN_DIR)

# Esecuzione della versione parallela (Default: N=2)
# Sintassi: make run NGRAM_SIZE=[2 o 3]
run: $(BIN_DIR)/$(TARGET)_par
	@./$(BIN_DIR)/$(TARGET)_par data/Texts $(NGRAM_SIZE)

# Esecuzione della versione sequenziale (Controllo)
run_seq: $(BIN_DIR)/$(TARGET)_seq
	@./$(BIN_DIR)/$(TARGET)_seq data/Texts ${NGRAM_SIZE:-2}

clean:
	rm -rf $(BIN_DIR)