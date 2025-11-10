"""
Visualize performance comparison between ACO and A* for TSP.
"""

import os
import sys
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

# Setup path
current_file = Path(__file__).resolve()
src_dir = current_file.parent.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


def load_performance_metrics(results_dir: str) -> Dict:
    """Load performance metrics from custom CSV format."""
    
    results_path = Path(results_dir)
    if not results_path.exists():
        return {}
    
    metrics = {}
    
    # Load CSV files (e.g., berlin52_metrics.csv)
    for csv_file in results_path.glob('*_metrics.csv'):
        problem_name = csv_file.stem.replace('_metrics', '')
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Map your CSV column names to standard metric keys
                algorithm = row.get('Algorithm', '').strip()
                if not algorithm:
                    continue

                # Initialize structure
                if algorithm not in metrics:
                    metrics[algorithm] = {
                        'problems': [],
                        'best_fitness': [],
                        'mean_fitness': [],
                        'std_fitness': [],
                        'mean_time': [],
                        'mean_memory': []
                    }

                # Extract metrics safely
                try:
                    avg_conv_speed = float(row.get('Avg Convergence Speed', 0))       # dùng làm "mean_time"
                    memory_usage = float(row.get('Memory Usage', 0)) / 1024           # KB -> MB
                    avg_tour_len = float(row.get('Avg Tour Length', 0))
                except ValueError:
                    avg_conv_speed, memory_usage, avg_tour_len = 0, 0, 0

                metrics[algorithm]['problems'].append(problem_name)
                metrics[algorithm]['best_fitness'].append(avg_tour_len)
                metrics[algorithm]['mean_fitness'].append(float(row.get('Avg Tour Length', 0)))
                metrics[algorithm]['mean_time'].append(float(row.get('Avg Convergence Speed', 0)))
                metrics[algorithm]['mean_memory'].append(float(row.get('Memory Usage', 0)))
                metrics[algorithm]['std_fitness'].append(0)  # vì file của bạn không có cột std
    
    return metrics

def plot_fitness_comparison(metrics: Dict, output_dir: str):
    """Plot fitness comparison between algorithms."""
    if not metrics:
        return
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    algorithms = list(metrics.keys())
    problems = metrics[algorithms[0]]['problems'] if algorithms else []
    if not problems:
        return
    x = np.arange(len(problems))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, alg in enumerate(algorithms):
        mean_fit = metrics[alg]['mean_fitness']
        std_fit = metrics[alg]['std_fitness']
        offset = width * (i - len(algorithms)/2 + 0.5)
        ax.bar(x + offset, mean_fit, width, yerr=std_fit,
               label=alg, capsize=5, alpha=0.8)
    ax.set_xlabel('Problem', fontsize=12)
    ax.set_ylabel('Tour Length (Mean ± Std)', fontsize=12)
    ax.set_title('Algorithm Performance Comparison - Tour Quality',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace('_', ' ').title() for p in problems])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    save_path = output_path / 'fitness_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_time_comparison(metrics: Dict, output_dir: str):
    """Plot computation time comparison."""
    if not metrics:
        return
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    algorithms = list(metrics.keys())
    problems = metrics[algorithms[0]]['problems'] if algorithms else []
    if not problems:
        return
    x = np.arange(len(problems))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, alg in enumerate(algorithms):
        mean_time = metrics[alg]['mean_time']
        offset = width * (i - len(algorithms)/2 + 0.5)
        ax.bar(x + offset, mean_time, width, label=alg, alpha=0.8)
    ax.set_xlabel('Problem', fontsize=12)
    ax.set_ylabel('Convergence Speed (arbitrary units)', fontsize=12)
    ax.set_title('Algorithm Comparison - Average Convergence Speed',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace('_', ' ').title() for p in problems])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    save_path = output_path / 'time_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_memory_comparison(metrics: Dict, output_dir: str):
    """Plot memory usage comparison."""
    if not metrics:
        return
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    algorithms = list(metrics.keys())
    problems = metrics[algorithms[0]]['problems'] if algorithms else []
    if not problems:
        return
    x = np.arange(len(problems))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, alg in enumerate(algorithms):
        mean_memory = metrics[alg]['mean_memory']
        offset = width * (i - len(algorithms)/2 + 0.5)
        ax.bar(x + offset, mean_memory, width, label=alg, alpha=0.8)
    ax.set_xlabel('Problem', fontsize=12)
    ax.set_ylabel('Memory Usage (MB)', fontsize=12)
    ax.set_title('Algorithm Performance Comparison - Memory Usage',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace('_', ' ').title() for p in problems])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    save_path = output_path / 'memory_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def visualize_performance_comparison(
    input_dir: str = 'results/discrete/performance',
    output_dir: str = 'visualizations/discrete/performance'
):
    """Main function to visualize performance comparison."""
    print("\n" + "="*80)
    print(" Visualizing Performance Comparison - Discrete Algorithms")
    print("="*80)
    print("\n  Loading performance metrics...")
    metrics = load_performance_metrics(input_dir)
    if not metrics:
        print(f"  No metrics found in {input_dir}")
        print("  Make sure *_metrics.csv files exist.")
        return
    print(f"  Found metrics for {len(metrics)} algorithms")
    print("\n  Creating visualizations...")
    plot_fitness_comparison(metrics, output_dir)
    plot_time_comparison(metrics, output_dir)
    plot_memory_comparison(metrics, output_dir)
    print("\n" + "="*80)
    print(" Performance comparison visualization completed!")
    print(f" Output directory: {output_dir}")
    print("="*80 + "\n")
