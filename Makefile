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
MODE ?= SCALING

.PHONY: all seq par clean run run_all scaling workload

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

run_all: scaling workload

scaling: $(BIN_DIR)/$(TARGET)_par
	@echo "=== Running SCALING test ==="
	./$(BIN_DIR)/$(TARGET)_par $(DATA_DIR) $(NGRAM_SIZE) $(NTHREADS) SCALING

workload: $(BIN_DIR)/$(TARGET)_par
	@echo "=== Running WORKLOAD test ==="
	./$(BIN_DIR)/$(TARGET)_par $(DATA_DIR) $(NGRAM_SIZE) $(NTHREADS) WORKLOAD

plot_scaling:
	@echo "Generazione grafici scaling..."
	python3 plot_results.py results/scaling_$(NGRAM_SIZE)gram.csv

plot_workload:
	@echo "Generazione grafici workload.."
	python3 plot_results.py results/workload_$(NGRAM_SIZE)gram$(NTHREADS).csv

plot_all:
	@echo "Generazione di tutti i grafici.."
	@for file in results/*.csv;
	do @echo "Processing $$file...";
	python3 plot_results.py $$file;
	done

clean:
	rm -rf $(BIN_DIR)