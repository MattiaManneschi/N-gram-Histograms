# Makefile per Progetto Mid-Term (Bigrammi/Trigrammi)

CXX = g++
CXXFLAGS = -std=c++17 -Wall -O3
CXXFLAGS_DEBUG = -std=c++17 -Wall -O3 -g  # Con simboli debug per profiling
OPENMP_FLAG = -fopenmp

TARGET = ngram_analyzer
SRC_DIR = src
BIN_DIR = bin
SRCS = $(wildcard $(SRC_DIR)/*.cpp)
NTHREADS ?= 16
NGRAM_SIZE ?= 2
DATA_DIR ?= data/Texts
MODE ?= THREAD

.PHONY: all seq par clean run run_all thread workload profile profile_stat profile_record profile_cache

all: $(BIN_DIR)/$(TARGET)_par

# =============================================================================
# COMPILAZIONE
# =============================================================================

$(BIN_DIR)/$(TARGET)_par: $(SRCS) | $(BIN_DIR)
	@echo "Compiling PARALLEL version..."
	$(CXX) $(CXXFLAGS) $(OPENMP_FLAG) $^ -o $@

$(BIN_DIR)/$(TARGET)_seq: $(SRCS) | $(BIN_DIR)
	@echo "Compiling SEQUENTIAL version..."
	$(CXX) $(CXXFLAGS) $^ -o $@

# Versione con simboli debug per profiling
$(BIN_DIR)/$(TARGET)_profile: $(SRCS) | $(BIN_DIR)
	@echo "Compiling PROFILE version (with debug symbols)..."
	$(CXX) $(CXXFLAGS_DEBUG) $(OPENMP_FLAG) $^ -o $@

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

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
# PROFILING CON PERF (Linux - funziona su AMD e Intel)
# =============================================================================

# Statistiche generali (CPU, IPC, cache)
profile_stat: $(BIN_DIR)/$(TARGET)_profile
	@echo "=== PERF STAT: Statistiche generali ==="
	perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses \
		./$(BIN_DIR)/$(TARGET)_profile $(DATA_DIR) $(NGRAM_SIZE) $(NTHREADS) $(MODE)

# Record + Report (hotspots)
profile_record: $(BIN_DIR)/$(TARGET)_profile
	@echo "=== PERF RECORD: Registrazione hotspots ==="
	perf record -g ./$(BIN_DIR)/$(TARGET)_profile $(DATA_DIR) $(NGRAM_SIZE) $(NTHREADS) $(MODE)
	@echo "=== PERF REPORT: Analisi ==="
	perf report

# Analisi cache dettagliata
profile_cache: $(BIN_DIR)/$(TARGET)_profile
	@echo "=== PERF STAT: Analisi Cache ==="
	perf stat -e L1-dcache-loads,L1-dcache-load-misses,LLC-loads,LLC-load-misses \
		./$(BIN_DIR)/$(TARGET)_profile $(DATA_DIR) $(NGRAM_SIZE) $(NTHREADS) $(MODE)

# Profiling completo (stat + record)
profile: profile_stat
	@echo ""
	@echo "Per analisi hotspots dettagliata, esegui: make profile_record"
	@echo "Per analisi cache dettagliata, esegui: make profile_cache"

# =============================================================================
# PROFILING CON AMD uProf (se installato)
# =============================================================================

UPROF_CLI = AMDuProfCLI
UPROF_OUTPUT = ./uprof_output

profile_uprof: $(BIN_DIR)/$(TARGET)_profile
	@echo "=== AMD uProf: Profiling ==="
	@mkdir -p $(UPROF_OUTPUT)
	$(UPROF_CLI) collect --config tbp -o $(UPROF_OUTPUT) \
		./$(BIN_DIR)/$(TARGET)_profile $(DATA_DIR) $(NGRAM_SIZE) $(NTHREADS) $(MODE)
	@echo "=== AMD uProf: Report ==="
	$(UPROF_CLI) report -i $(UPROF_OUTPUT)

# =============================================================================
# PROFILING CON VALGRIND/CALLGRIND (indipendente da CPU)
# =============================================================================

profile_callgrind: $(BIN_DIR)/$(TARGET)_profile
	@echo "=== CALLGRIND: Analisi call graph ==="
	@echo "NOTA: Valgrind è lento, usa NTHREADS=1 e NGRAM_SIZE=2"
	valgrind --tool=callgrind --callgrind-out-file=callgrind.out \
		./$(BIN_DIR)/$(TARGET)_profile $(DATA_DIR) 2 1 $(MODE)
	@echo "Visualizza con: kcachegrind callgrind.out"

# =============================================================================
# CLEAN
# =============================================================================

clean:
	rm -rf $(BIN_DIR)
	rm -f perf.data perf.data.old callgrind.out*
	rm -rf $(UPROF_OUTPUT)