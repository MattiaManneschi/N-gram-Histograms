# Makefile per Progetto Mid-Term (Bigrammi/Trigrammi)

# Rileva sistema operativo
UNAME_S := $(shell uname -s)

CXX = g++
CXXFLAGS = -std=c++17 -Wall -O3

# OpenMP flags per diversi sistemi
ifeq ($(UNAME_S),Darwin)
    # macOS con Homebrew libomp
    OPENMP_FLAG = -Xpreprocessor -fopenmp
    OPENMP_LIBS = -lomp
    # Percorsi Homebrew (Intel Mac)
    HOMEBREW_PREFIX := $(shell brew --prefix 2>/dev/null || echo /usr/local)
    CXXFLAGS += -I$(HOMEBREW_PREFIX)/opt/libomp/include
    LDFLAGS += -L$(HOMEBREW_PREFIX)/opt/libomp/lib
else
    # Linux
    OPENMP_FLAG = -fopenmp
    OPENMP_LIBS =
    LDFLAGS =
endif

TARGET = ngram_analyzer
SRC_DIR = src
BIN_DIR = bin
SRCS = $(wildcard $(SRC_DIR)/*.cpp)
NTHREADS ?= 16
NGRAM_SIZE ?= 2
DATA_DIR ?= data/Texts
MODE ?= THREAD
MULTIPLIER ?= 1

.PHONY: all seq par clean run run_all thread workload stats

all: $(BIN_DIR)/$(TARGET)_par

# =============================================================================
# COMPILAZIONE BENCHMARK
# =============================================================================

$(BIN_DIR)/$(TARGET)_par: $(SRCS) | $(BIN_DIR)
	@echo "Compiling PARALLEL version..."
	@echo "System: $(UNAME_S)"
	$(CXX) $(CXXFLAGS) $(OPENMP_FLAG) $^ -o $@ $(LDFLAGS) $(OPENMP_LIBS)

$(BIN_DIR)/$(TARGET)_seq: $(SRCS) | $(BIN_DIR)
	@echo "Compiling SEQUENTIAL version..."
	$(CXX) $(CXXFLAGS) $^ -o $@

# =============================================================================
# COMPILAZIONE STATISTICS
# =============================================================================

# Sorgenti per statistics (escludi main_driver.cpp)
STATS_SRCS = $(SRC_DIR)/data_loader.cpp \
             $(SRC_DIR)/ngram_counter_stats.cpp \
             $(SRC_DIR)/statistics.cpp \
             $(SRC_DIR)/run_statistics.cpp

$(BIN_DIR)/run_statistics: $(STATS_SRCS) | $(BIN_DIR)
	@echo "Compiling STATISTICS version..."
	$(CXX) $(CXXFLAGS) $(OPENMP_FLAG) $^ -o $@ $(LDFLAGS) $(OPENMP_LIBS)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)
	mkdir -p results/statistics

# =============================================================================
# ESECUZIONE BENCHMARK
# =============================================================================

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

# =============================================================================
# STATISTICHE
# =============================================================================

stats: $(BIN_DIR)/run_statistics
	@echo "=== Generating N-gram Statistics ==="
	./$(BIN_DIR)/run_statistics $(DATA_DIR) $(NGRAM_SIZE) $(NTHREADS) $(MULTIPLIER)
	@echo "=== Generating Zipf Plot ==="
	python3 plot_statistics.py $(NGRAM_SIZE) $(MULTIPLIER)

stats_all: $(BIN_DIR)/run_statistics
	@echo "=== Generating Statistics for 2-grams ==="
	./$(BIN_DIR)/run_statistics $(DATA_DIR) 2 $(NTHREADS) 1
	@echo "=== Generating Statistics for 3-grams ==="
	./$(BIN_DIR)/run_statistics $(DATA_DIR) 3 $(NTHREADS) 1
	@echo "=== Generating Plots ==="
	python3 plot_statistics.py 2 1
	python3 plot_statistics.py 3 1

# =============================================================================
# CLEAN
# =============================================================================

clean:
	rm -rf $(BIN_DIR)
	rm -f results/*.csv results/*.png results/*.txt
	rm -rf results/statistics