#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

matplotlib.use('Agg')

# Cartelle e file CSV
ngram_size = int(sys.argv[1])
threads = int(sys.argv[2])

scaling_csv = f"results/thread_scaling_{ngram_size}gram.csv"
workload_csv = f"results/workload_{ngram_size}gram_t{threads}.csv"
OUTPUT_DIR = "results"

# Colori e marker per le strategie
STRATEGY_STYLES = {
    "Hybrid-TLS":   {"color": "tab:blue",   "marker": "o", "linestyle": "-"},
    "Document-level-TLS":  {"color": "tab:purple",   "marker": "o", "linestyle": "dotted"},
    "Fine-grained-locking": {"color": "tab:green",  "marker": "x", "linestyle": ":"},
}

def plot_speedup(df, x_col, title, filename, x_label):
    plt.figure(figsize=(10,6))
    strategies = df['Strategy'].unique()

    for strategy in strategies:
        strat_df = df[df['Strategy'] == strategy].sort_values(x_col)
        style = STRATEGY_STYLES.get(strategy, {"color": "black", "marker": "o", "linestyle": "-"})
        plt.plot(strat_df[x_col], strat_df['Speedup'],
                 color=style["color"], marker=style["marker"], linestyle=style["linestyle"],
                 label=strategy)

    plt.xlabel(x_label)
    plt.ylabel("Speedup")
    plt.title(title)
    plt.grid(True)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    filepath = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(filepath)
    plt.close()
    print(f"✓ Plot Speedup salvato in: {filepath}")


if os.path.exists(scaling_csv):
    df_scaling = pd.read_csv(scaling_csv)
    plot_speedup(df_scaling, x_col='Threads',
                 title='Scaling Test: Speedup vs Threads',
                 filename='scaling_speedup.png',
                 x_label='Number of Threads')
else:
    print(f"⚠ File {scaling_csv} non trovato, salto il test scaling.")


if os.path.exists(workload_csv):
    df_workload = pd.read_csv(workload_csv)
    # Non filtriamo più per fixed threads
    plot_speedup(df_workload, x_col='Multiplier',
                 title='Workload Test: Speedup vs Workload',
                 filename='workload_speedup.png',
                 x_label='Workload Multiplier')
else:
    print(f"⚠ File {workload_csv} non trovato, salto il test workload.")

print("✓ Tutti i plot speedup completati.")
