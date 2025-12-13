import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from pathlib import Path

# Stile grafici
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

def plot_scaling_results(csv_file):
    """Genera grafici per strong scaling test"""

    df = pd.read_csv(csv_file)
    base_name = Path(csv_file).stem
    output_dir = Path(csv_file).parent / "plots"
    output_dir.mkdir(exist_ok=True)

    strategies = df['Strategy'].unique()

    # 1. SPEEDUP vs THREADS
    plt.figure(figsize=(12, 7))
    for strategy in strategies:
        data = df[df['Strategy'] == strategy]
        plt.plot(data['Threads'], data['Speedup'],
                 marker='o', linewidth=2, markersize=8, label=strategy)

    max_threads = df['Threads'].max()
    plt.plot([1, max_threads], [1, max_threads],
             'k--', alpha=0.5, linewidth=2, label='Ideal (Linear)')

    plt.xlabel('Numero di Thread', fontsize=13, fontweight='bold')
    plt.ylabel('Speedup', fontsize=13, fontweight='bold')
    plt.title('Strong Scaling: Speedup vs Threads', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11, loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{base_name}_speedup.png", dpi=300, bbox_inches='tight')
    print(f"✓ Salvato: {output_dir / f'{base_name}_speedup.png'}")
    plt.close()

    # 2. EFFICIENCY vs THREADS
    plt.figure(figsize=(12, 7))
    for strategy in strategies:
        data = df[df['Strategy'] == strategy]
        plt.plot(data['Threads'], data['Efficiency_percent'],
                 marker='s', linewidth=2, markersize=8, label=strategy)

    plt.axhline(y=100, color='k', linestyle='--', alpha=0.5, linewidth=2, label='Ideal (100%)')
    plt.xlabel('Numero di Thread', fontsize=13, fontweight='bold')
    plt.ylabel('Efficienza (%)', fontsize=13, fontweight='bold')
    plt.title('Strong Scaling: Efficienza vs Threads', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 110)
    plt.tight_layout()
    plt.savefig(output_dir / f"{base_name}_efficiency.png", dpi=300, bbox_inches='tight')
    print(f"✓ Salvato: {output_dir / f'{base_name}_efficiency.png'}")
    plt.close()

    # 3. EXECUTION TIME vs THREADS (log scale)
    plt.figure(figsize=(12, 7))
    for strategy in strategies:
        data = df[df['Strategy'] == strategy]
        plt.semilogy(data['Threads'], data['Time_seconds'],
                     marker='D', linewidth=2, markersize=8, label=strategy)

    plt.xlabel('Numero di Thread', fontsize=13, fontweight='bold')
    plt.ylabel('Tempo di Esecuzione (secondi, log scale)', fontsize=13, fontweight='bold')
    plt.title('Strong Scaling: Tempo di Esecuzione vs Threads', fontsize=15, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(output_dir / f"{base_name}_time.png", dpi=300, bbox_inches='tight')
    print(f"✓ Salvato: {output_dir / f'{base_name}_time.png'}")
    plt.close()

    # 4. BAR CHART COMPARATIVO (a max threads)
    max_thread_data = df[df['Threads'] == max_threads]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.bar(max_thread_data['Strategy'], max_thread_data['Speedup'],
            color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])
    ax1.axhline(y=max_threads, color='k', linestyle='--', alpha=0.5, label=f'Ideal ({max_threads}x)')
    ax1.set_ylabel('Speedup', fontsize=12, fontweight='bold')
    ax1.set_title(f'Confronto Speedup ({max_threads} threads)', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=15, ha='right')

    ax2.bar(max_thread_data['Strategy'], max_thread_data['Efficiency_percent'],
            color=['#2ecc71', '#3498db', '#e74c3c', '#f39c12'])
    ax2.axhline(y=100, color='k', linestyle='--', alpha=0.5, label='Ideal (100%)')
    ax2.set_ylabel('Efficienza (%)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Confronto Efficienza ({max_threads} threads)', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(0, 110)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=15, ha='right')

    plt.tight_layout()
    plt.savefig(output_dir / f"{base_name}_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✓ Salvato: {output_dir / f'{base_name}_comparison.png'}")
    plt.close()

def plot_workload_results(csv_file):
    """Genera grafici per workload scaling test"""

    df = pd.read_csv(csv_file)
    base_name = Path(csv_file).stem
    output_dir = Path(csv_file).parent / "plots"
    output_dir.mkdir(exist_ok=True)

    strategies = df['Strategy'].unique()
    fixed_threads = df['Threads'].iloc[0]

    # 1. SPEEDUP vs WORKLOAD MULTIPLIER
    plt.figure(figsize=(12, 7))
    for strategy in strategies:
        data = df[df['Strategy'] == strategy]
        plt.plot(data['Multiplier'], data['Speedup'],
                 marker='o', linewidth=2, markersize=8, label=strategy)

    plt.xlabel('Workload Multiplier', fontsize=13, fontweight='bold')
    plt.ylabel('Speedup', fontsize=13, fontweight='bold')
    plt.title(f'Workload Scaling: Speedup vs Workload ({fixed_threads} threads)',
              fontsize=15, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{base_name}_speedup.png", dpi=300, bbox_inches='tight')
    print(f"✓ Salvato: {output_dir / f'{base_name}_speedup.png'}")
    plt.close()

    # 2. EXECUTION TIME vs WORKLOAD
    plt.figure(figsize=(12, 7))
    for strategy in strategies:
        data = df[df['Strategy'] == strategy]
        plt.plot(data['Multiplier'], data['Time_seconds'],
                 marker='s', linewidth=2, markersize=8, label=strategy)

    plt.xlabel('Workload Multiplier', fontsize=13, fontweight='bold')
    plt.ylabel('Tempo di Esecuzione (secondi)', fontsize=13, fontweight='bold')
    plt.title(f'Workload Scaling: Tempo vs Workload ({fixed_threads} threads)',
              fontsize=15, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{base_name}_time.png", dpi=300, bbox_inches='tight')
    print(f"✓ Salvato: {output_dir / f'{base_name}_time.png'}")
    plt.close()

    # 3. EFFICIENCY vs WORKLOAD
    plt.figure(figsize=(12, 7))
    for strategy in strategies:
        data = df[df['Strategy'] == strategy]
        plt.plot(data['Multiplier'], data['Efficiency_percent'],
                 marker='D', linewidth=2, markersize=8, label=strategy)

    plt.axhline(y=100, color='k', linestyle='--', alpha=0.5, linewidth=2, label='Ideal')
    plt.xlabel('Workload Multiplier', fontsize=13, fontweight='bold')
    plt.ylabel('Efficienza (%)', fontsize=13, fontweight='bold')
    plt.title(f'Workload Scaling: Efficienza vs Workload ({fixed_threads} threads)',
              fontsize=15, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"{base_name}_efficiency.png", dpi=300, bbox_inches='tight')
    print(f"✓ Salvato: {output_dir / f'{base_name}_efficiency.png'}")
    plt.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 plot_results.py <csv_file>")
        print("Example: python3 plot_results.py results/scaling_2gram.csv")
        sys.exit(1)

    csv_file = sys.argv[1]

    if not os.path.exists(csv_file):
        print(f"Errore: File {csv_file} non trovato!")
        sys.exit(1)

    print(f"\n📊 Generazione grafici da: {csv_file}\n")

    if 'scaling' in Path(csv_file).stem and 'workload' not in Path(csv_file).stem:
        plot_scaling_results(csv_file)
    elif 'workload' in Path(csv_file).stem:
        plot_workload_results(csv_file)
    else:
        print("Errore: Tipo di test non riconosciuto dal nome del file")
        sys.exit(1)

    print("\n✅ Grafici generati con successo!\n")

if __name__ == "__main__":
    main()