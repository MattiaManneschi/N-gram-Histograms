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
MODE ?= THREAD

.PHONY: all seq par clean run run_all thread workload

all: $(BIN_DIR)/$(TARGET)_par

$(BIN_DIR)/$(TARGET)_par: $(SRCS) | $(BIN_DIR)
	@echo "Compiling PARALLEL version..."
	$(CXX) $(CXXFLAGS) $(OPENMP_FLAG) $^ -o $@

$(BIN_DIR)/$(TARGET)_seq: $(SRCS) | $(BIN_DIR)
	@echo "Compiling SEQUENTIAL version..."
	$(CXX) $(CXXFLAGS) $^ -o $@

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

run: $(BIN_DIR)/$(TARGET)_par
	@echo "Esecuzione: NGRAM_SIZE=$(NGRAM_SIZE), NTHREADS=$(NTHREADS), MODE=$(MODE)"
	./$(BIN_DIR)/$(TARGET)_par $(DATA_DIR) $(NGRAM_SIZE) $(NTHREADS) $(MODE)
	python3 plot_results.py $(NGRAM_SIZE) $(NTHREADS)

run_all: thread workload

thread: $(BIN_DIR)/$(TARGET)_par
	@echo "=== Running THREAD SCALING test ==="
	./$(BIN_DIR)/$(TARGET)_par $(DATA_DIR) $(NGRAM_SIZE) $(NTHREADS) THREAD
	@echo "=== Generazione grafici scaling ==="
	python3 plot_results.py $(NGRAM_SIZE) $(NTHREADS)

workload: $(BIN_DIR)/$(TARGET)_par
	@echo "=== Running WORKLOAD SCALING test ==="
	./$(BIN_DIR)/$(TARGET)_par $(DATA_DIR) $(NGRAM_SIZE) $(NTHREADS) WORKLOAD
	@echo "=== Generazione grafici workload ==="
	python3 plot_results.py $(NGRAM_SIZE) $(NTHREADS)

clean:
	rm -rf $(BIN_DIR)