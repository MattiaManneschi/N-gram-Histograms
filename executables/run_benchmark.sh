





if [ ! -t 0 ]; then
    
    if command -v gnome-terminal &>/dev/null; then
        gnome-terminal -- bash -c "cd '$(pwd)' && '$0'; exec bash"
    elif command -v konsole &>/dev/null; then
        konsole -e bash -c "cd '$(pwd)' && '$0'; exec bash"
    elif command -v xfce4-terminal &>/dev/null; then
        xfce4-terminal -e "bash -c \"cd '$(pwd)' && '$0'; exec bash\""
    elif command -v xterm &>/dev/null; then
        xterm -e "bash -c \"cd '$(pwd)' && '$0'; exec bash\""
    else
        echo "Nessun terminale trovato. Esegui da terminale: bash $0"
    fi
    exit 0
fi





RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'


REPO_URL="https://github.com/MattiaManneschi/N-gram-Histograms.git"
REPO_DIR="N-gram-Histograms"
NTHREADS=$(nproc)





print_header() {
    echo -e "${CYAN}"
    echo "╔═══════════════════════════════════════════════════════════════╗"
    echo "║         N-GRAM HISTOGRAM BENCHMARK RUNNER (Linux)             ║"
    echo "╚═══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

get_timestamp() {
    date +"%Y%m%d_%H%M%S"
}





setup_working_directory() {
    print_step "Rilevamento directory di lavoro..."

    
    if [ -d "src" ] && [ -f "Makefile" ] && [ -d "data/Texts" ]; then
        print_success "Già dentro il repository"
        WORK_DIR="$(pwd)"
    
    elif [ -d "$REPO_DIR/src" ] && [ -f "$REPO_DIR/Makefile" ]; then
        print_success "Repository trovato in ./$REPO_DIR"
        cd "$REPO_DIR"
        WORK_DIR="$(pwd)"
    else
        
        print_step "Repository non trovato, clonazione..."
        git clone "$REPO_URL"
        cd "$REPO_DIR"
        WORK_DIR="$(pwd)"
        print_success "Repository clonato"
    fi

    echo "Working directory: $WORK_DIR"
}





install_dependencies() {
    print_step "Controllo dipendenze..."

    local missing=()

    command -v g++ &>/dev/null || missing+=("g++")
    command -v make &>/dev/null || missing+=("make")
    command -v python3 &>/dev/null || missing+=("python3")

    if [ ${#missing[@]} -eq 0 ]; then
        print_success "Tutte le dipendenze sono installate"
    else
        print_warning "Dipendenze mancanti: ${missing[*]}"
        print_step "Tentativo di installazione..."

        if command -v apt-get &>/dev/null; then
            sudo apt-get update
            sudo apt-get install -y build-essential python3 python3-pip
        elif command -v dnf &>/dev/null; then
            sudo dnf install -y gcc-c++ make python3 python3-pip
        elif command -v pacman &>/dev/null; then
            sudo pacman -S --noconfirm gcc make python python-pip
        else
            print_error "Installa manualmente: ${missing[*]}"
            exit 1
        fi
    fi

    
    print_step "Installazione pacchetti Python..."
    pip3 install --user matplotlib pandas numpy 2>/dev/null || \
    pip install --user matplotlib pandas numpy 2>/dev/null || \
    python3 -m pip install --user matplotlib pandas numpy || true
    print_success "Pacchetti Python OK"

    echo "DEBUG: Fine install_dependencies"
}





compile_project() {
    print_step "Compilazione progetto..."
    echo "DEBUG: Inizio compilazione in $(pwd)"
    make clean 2>/dev/null || true
    echo "DEBUG: make clean OK"
    make all
    echo "DEBUG: make all OK"
    print_success "Compilazione completata"
}





select_ngram_size() {
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  Seleziona la dimensione degli N-grammi:${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo "  1) 2-grammi (bigrammi)"
    echo "  2) 3-grammi (trigrammi)"
    echo ""

    
    if [ -t 0 ]; then
        read -p "Scelta [1-2]: " choice
    else
        echo "ERRORE: Script deve essere eseguito in un terminale interattivo"
        echo "Uso: bash ./run_benchmark.sh"
        exit 1
    fi

    case $choice in
        1) NGRAM_SIZE=2 ;;
        2) NGRAM_SIZE=3 ;;
        *) NGRAM_SIZE=2 ;;
    esac

    print_success "Selezionato: $NGRAM_SIZE-grammi"
}

select_test_mode() {
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  Seleziona il tipo di test:${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo "  1) Thread Scaling  (workload fisso, thread variabili)"
    echo "  2) Workload Scaling (thread fissi, workload variabile)"
    echo "  3) Entrambi"
    echo ""
    read -p "Scelta [1-3]: " choice

    case $choice in
        1) TEST_MODE="THREAD" ;;
        2) TEST_MODE="WORKLOAD" ;;
        3) TEST_MODE="BOTH" ;;
        *) TEST_MODE="THREAD" ;;
    esac

    print_success "Selezionato: $TEST_MODE"
}

select_threads() {
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  Numero di thread (rilevati: $NTHREADS core):${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    read -p "Numero thread [$NTHREADS]: " input_threads

    if [ -n "$input_threads" ]; then
        NTHREADS=$input_threads
    fi

    print_success "Selezionato: $NTHREADS thread"
}





run_test() {
    local mode=$1
    local timestamp=$(get_timestamp)

    
    local output_dir="results_${NGRAM_SIZE}gram/${mode,,}_${timestamp}"

    print_step "Esecuzione test $mode per $NGRAM_SIZE-grammi..."
    print_step "Output: $output_dir"

    mkdir -p "$output_dir"

    
    ./bin/ngram_analyzer_par data/Texts $NGRAM_SIZE $NTHREADS $mode

    
    python3 plot_results.py $NGRAM_SIZE $NTHREADS || python plot_results.py $NGRAM_SIZE $NTHREADS

    
    if [ -d "results" ]; then
        cp -r results/* "$output_dir/" 2>/dev/null || true
    fi

    
    cat > "$output_dir/test_info.txt" << EOF
═══════════════════════════════════════════════════════════════
  N-GRAM BENCHMARK TEST INFO
═══════════════════════════════════════════════════════════════

Data/Ora:       $(date)
N-gram size:    $NGRAM_SIZE
Test mode:      $mode
Threads:        $NTHREADS
CPU:            $(grep "model name" /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)
Core:           $(nproc)

═══════════════════════════════════════════════════════════════
EOF

    print_success "Risultati salvati in: $output_dir"
}





main() {
    print_header

    echo "DEBUG: Inizio main"

    
    setup_working_directory
    echo "DEBUG: Dopo setup_working_directory"

    install_dependencies
    echo "DEBUG: Dopo install_dependencies"

    compile_project
    echo "DEBUG: Dopo compile_project"

    
    select_ngram_size
    select_test_mode
    select_threads

    
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  AVVIO BENCHMARK${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"

    case $TEST_MODE in
        THREAD)   run_test "THREAD" ;;
        WORKLOAD) run_test "WORKLOAD" ;;
        BOTH)     run_test "THREAD"; run_test "WORKLOAD" ;;
    esac

    
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  BENCHMARK COMPLETATO!${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo "Risultati in: results_${NGRAM_SIZE}gram/"
    ls -la "results_${NGRAM_SIZE}gram/" 2>/dev/null || true
}

main "$@"