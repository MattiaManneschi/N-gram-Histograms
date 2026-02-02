#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_statistics.py - Plot per Statistiche N-gram (Zipf's Law)

Uso:
  python plot_statistics.py <ngram_size>

Esempio:
  python plot_statistics.py 2
"""

import matplotlib

matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10

STATS_DIR = "../results/statistics"


def plot_zipf(ngram_size):
    """Genera Zipf plot (log-log)"""

    freq_csv = f"{STATS_DIR}/{ngram_size}gram_freq_dist.csv"
    output_path = f"{STATS_DIR}/{ngram_size}gram_zipf.png"

    if not os.path.exists(freq_csv):
        print(f"File not found: {freq_csv}")
        return False

    df = pd.read_csv(freq_csv)

    frequencies = []
    for _, row in df.iterrows():
        frequencies.extend([row['frequency']] * row['count'])

    frequencies.sort(reverse=True)
    ranks = range(1, len(frequencies) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.loglog(ranks, frequencies, 'b-', linewidth=0.5, alpha=0.7)
    ax1.set_xlabel("Rank (log scale)", fontsize=12)
    ax1.set_ylabel("Frequency (log scale)", fontsize=12)
    ax1.set_title(f"Zipf's Law Plot - {ngram_size}-grams", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, which='both')

    if len(frequencies) > 0:
        C = frequencies[0]
        theoretical = [C / r for r in ranks[:min(10000, len(ranks))]]
        ax1.loglog(ranks[:len(theoretical)], theoretical, 'r--',
                   linewidth=1, alpha=0.5, label='Theoretical Zipf (1/r)')
        ax1.legend()

    df_sorted = df.sort_values('frequency')
    ax2.bar(df_sorted['frequency'][:30], df_sorted['count'][:30],
            color='steelblue', alpha=0.7)
    ax2.set_xlabel("Frequency", fontsize=12)
    ax2.set_ylabel("Number of N-grams", fontsize=12)
    ax2.set_title(f"Frequency Distribution - {ngram_size}-grams",
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")
    return True


def plot_top_ngrams(ngram_size, top_n=20):
    """Genera bar chart dei top n-grammi"""

    top_csv = f"{STATS_DIR}/{ngram_size}gram_top50.csv"
    output_path = f"{STATS_DIR}/{ngram_size}gram_top{top_n}.png"

    if not os.path.exists(top_csv):
        print(f"File not found: {top_csv}")
        return False

    df = pd.read_csv(top_csv).head(top_n)

    fig, ax = plt.subplots(figsize=(12, 8))

    y_pos = np.arange(len(df))
    bars = ax.barh(y_pos, df['frequency'], color='steelblue', alpha=0.8)

    labels = []
    for ng in df['ngram']:
        ng_str = str(ng)
        if len(ng_str) > 25:
            labels.append(f'"{ng_str[:22]}..."')
        else:
            labels.append(f'"{ng_str}"')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Frequency", fontsize=12)
    ax.set_title(f"Top {top_n} Most Frequent {ngram_size}-grams",
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')

    max_freq = df['frequency'].max()
    for bar, freq in zip(bars, df['frequency']):
        ax.text(bar.get_width() + max_freq * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{freq:,}', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")
    return True


def plot_summary(ngram_size):
    """Genera summary visuale delle statistiche"""

    stats_csv = f"{STATS_DIR}/{ngram_size}gram_stats.csv"
    output_path = f"{STATS_DIR}/{ngram_size}gram_summary.png"

    if not os.path.exists(stats_csv):
        print(f"File not found: {stats_csv}")
        return False

    df = pd.read_csv(stats_csv)
    stats = dict(zip(df['metric'], df['value']))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax1 = axes[0, 0]
    metrics = ['Total Tokens', 'Unique N-grams', 'Hapax', 'Dis Legomena']
    values = [
        int(float(stats.get('total_ngrams', 0))),
        int(float(stats.get('unique_ngrams', 0))),
        int(float(stats.get('hapax_legomena', 0))),
        int(float(stats.get('dis_legomena', 0)))
    ]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
    bars = ax1.bar(metrics, values, color=colors)
    ax1.set_ylabel('Count')
    ax1.set_title('Key Statistics', fontweight='bold')
    ax1.tick_params(axis='x', rotation=15)
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{val:,}', ha='center', va='bottom', fontsize=9)

    ax2 = axes[0, 1]
    hapax_ratio = float(stats.get('hapax_ratio', 0)) * 100
    coverage = float(stats.get('coverage_top_100', 0))

    categories = ['Hapax Ratio', 'Top-100 Coverage']
    percentages = [hapax_ratio, coverage]
    colors = ['#e74c3c', '#9b59b6']

    bars = ax2.bar(categories, percentages, color=colors)
    ax2.set_ylabel('Percentage (%)')
    ax2.set_title('Distribution Metrics', fontweight='bold')
    ax2.set_ylim(0, 100)
    for bar, pct in zip(bars, percentages):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{pct:.1f}%', ha='center', va='bottom', fontsize=10)

    ax3 = axes[1, 0]
    speedup = float(stats.get('speedup', 1))
    efficiency = float(stats.get('efficiency', 0)) * 100

    perf_metrics = ['Speedup', 'Efficiency (%)']
    perf_values = [speedup, efficiency]
    bars = ax3.bar(perf_metrics, perf_values, color=['#1abc9c', '#3498db'])
    ax3.set_title('Parallel Performance', fontweight='bold')
    for bar, val in zip(bars, perf_values):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f'{val:.2f}', ha='center', va='bottom', fontsize=10)

    ax4 = axes[1, 1]
    ax4.axis('off')

    info_text = f"""
    N-gram Statistics Summary
    
    N-gram size:       {ngram_size}
    Total tokens:      {int(float(stats.get('total_ngrams', 0))):,}
    Vocabulary size:   {int(float(stats.get('unique_ngrams', 0))):,}
    
    Max frequency:     {int(float(stats.get('max_frequency', 0))):,}
    Mean frequency:    {float(stats.get('mean_frequency', 0)):.2f}
    
    Zipf's Law Analysis:
    Hapax ratio: {hapax_ratio:.1f}%
    (typical: 40-60% for natural language)
    """

    ax4.text(0.1, 0.9, info_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'{ngram_size}-gram Analysis Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_statistics.py <ngram_size>")
        sys.exit(1)

    ngram_size = int(sys.argv[1])

    print(f"\n=== Generating Statistics Plots for {ngram_size}-grams ===\n")

    plot_zipf(ngram_size)
    plot_top_ngrams(ngram_size)
    plot_summary(ngram_size)

    print(f"\nAll plots saved in {STATS_DIR}/")


if __name__ == "__main__":
    main()
