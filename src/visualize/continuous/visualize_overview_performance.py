"""
This module creates visualizations comparing algorithms across multiple metrics:
- Convergence speed
- Computational complexity (time)
- Robustness
- Scalability

Results are saved to visualizations/performance/overview/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import sys
from pathlib import Path

# Setup matplotlib style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12


def load_metrics(problem_name, results_dir='results/continuous/performance'):
    """
    Load performance metrics from CSV file.
    
    Args:
        problem_name: Name of problem (sphere, ackley, rastrigin)
        results_dir: Directory containing results
    
    Returns:
        DataFrame with metrics
    """
    csv_file = Path(results_dir) / f'{problem_name.lower()}_metrics.csv'
    
    if not csv_file.exists():
        print(f" Metrics file not found: {csv_file}")
        return None
    
    return pd.read_csv(csv_file)


def load_scalability(problem_name, results_dir='results/continuous/performance'):
    """
    Load scalability data from JSON file.
    
    Args:
        problem_name: Name of problem
        results_dir: Directory containing results
    
    Returns:
        Dictionary with scalability data
    """
    json_file = Path(results_dir) / f'{problem_name.lower()}_scalability_data.json'
    
    if not json_file.exists():
        print(f" Scalability file not found: {json_file}")
        return None
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    return data


def plot_performance_comparison(problems=['sphere', 'ackley', 'rastrigin'],
                                results_dir='results/continuous/performance',
                                output_dir='visualizations/continuous/performance/overview'):
    """
    Create bar chart comparing algorithm performance across problems.
    
    Args:
        problems: List of problem names
        results_dir: Directory containing results
        output_dir: Directory to save visualization
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load all metrics
    all_data = []
    for problem_name in problems:
        df = load_metrics(problem_name, results_dir)
        if df is not None:
            df['Problem'] = problem_name.title()
            all_data.append(df)
    
    if not all_data:
        print(" No data available")
        return
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Color palette
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    
    # 1. Best Fitness comparison
    ax = axes[0, 0]
    pivot_df = combined_df.pivot(index='Algorithm', columns='Problem', values='Best Fitness')
    pivot_df.plot(kind='bar', ax=ax, color=colors[:len(problems)], alpha=0.8, width=0.7)
    ax.set_ylabel('Best Fitness (log scale)', fontweight='bold')
    ax.set_title('Best Fitness Comparison', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(title='Problem', framealpha=0.9)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 2. Convergence Speed comparison
    ax = axes[0, 1]
    pivot_df = combined_df.pivot(index='Algorithm', columns='Problem', values='Convergence Speed (iter)')
    pivot_df.plot(kind='bar', ax=ax, color=colors[:len(problems)], alpha=0.8, width=0.7)
    ax.set_ylabel('Iterations to Converge', fontweight='bold')
    ax.set_title('Convergence Speed Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(title='Problem', framealpha=0.9)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 3. Execution Time comparison
    ax = axes[1, 0]
    pivot_df = combined_df.pivot(index='Algorithm', columns='Problem', values='Mean Time (s)')
    pivot_df.plot(kind='bar', ax=ax, color=colors[:len(problems)], alpha=0.8, width=0.7)
    ax.set_ylabel('Execution Time (seconds)', fontweight='bold')
    ax.set_title('Computational Time Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(title='Problem', framealpha=0.9)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 4. Robustness comparison
    ax = axes[1, 1]
    pivot_df = combined_df.pivot(index='Algorithm', columns='Problem', values='Robustness')
    pivot_df.plot(kind='bar', ax=ax, color=colors[:len(problems)], alpha=0.8, width=0.7)
    ax.set_ylabel('Robustness Score', fontweight='bold')
    ax.set_title('Robustness Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(title='Problem', framealpha=0.9)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.suptitle('Algorithm Performance Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save figure
    output_file = output_path / 'performance_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def plot_scalability_comparison(problems=['sphere', 'ackley', 'rastrigin'],
                                results_dir='results/continuous/performance',
                                output_dir='visualizations/continuous/performance/overview'):
    """
    Plot scalability comparison - time vs problem size.
    
    Args:
        problems: List of problem names
        results_dir: Directory containing results
        output_dir: Directory to save visualization
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, len(problems), figsize=(6 * len(problems), 6))
    
    if len(problems) == 1:
        axes = [axes]
    
    colors = {
        'GA': '#2E86AB',
        'FA': '#A23B72',
        'PSO': '#F18F01',
        'SA': '#C73E1D',
        'CS': '#6A994E'
    }
    
    for idx, problem_name in enumerate(problems):
        data = load_scalability(problem_name, results_dir)
        
        if data is None:
            continue
        
        ax = axes[idx]
        
        for alg_name, alg_data in data.items():
            dimensions = alg_data['dimensions']
            times = alg_data['times']
            
            color = colors.get(alg_name, 'gray')
            ax.plot(dimensions, times, marker='o', linewidth=2.5,
                   markersize=8, label=alg_name, color=color, alpha=0.9)
        
        ax.set_xlabel('Problem Dimensions', fontweight='bold')
        ax.set_ylabel('Execution Time (seconds)', fontweight='bold')
        ax.set_title(f'{problem_name.title()} Function', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', framealpha=0.9)
    
    plt.suptitle('Scalability Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    output_file = output_path / 'scalability_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def plot_radar_chart(problem_name, results_dir='results/continuous/performance',
                    output_dir='visualizations/continuous/performance/overview'):
    """
    Create radar chart comparing algorithms across multiple metrics.
    
    Args:
        problem_name: Name of problem
        results_dir: Directory containing results
        output_dir: Directory to save visualization
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    df = load_metrics(problem_name, results_dir)
    
    if df is None:
        return
    
    # Normalize metrics to [0, 1] range
    metrics = ['Best Fitness', 'Convergence Speed (iter)', 'Mean Time (s)', 'Robustness']
    
    # Create normalized dataframe (lower is better for most metrics, higher is better for Robustness)
    df_norm = df.copy()
    
    # Inverse normalization for metrics where lower is better
    for metric in ['Best Fitness', 'Convergence Speed (iter)', 'Mean Time (s)']:
        if metric in df_norm.columns:
            max_val = df_norm[metric].max()
            min_val = df_norm[metric].min()
            if max_val > min_val:
                df_norm[metric] = 1 - (df_norm[metric] - min_val) / (max_val - min_val)
            else:
                df_norm[metric] = 1.0
    
    # Setup radar chart
    categories = ['Quality', 'Speed', 'Time\nEfficiency', 'Robustness']
    N = len(categories)
    
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    colors = {
        'GA': '#2E86AB',
        'FA': '#A23B72',
        'PSO': '#F18F01',
        'SA': '#C73E1D',
        'CS': '#6A994E'
    }
    
    for idx, row in df_norm.iterrows():
        alg_name = row['Algorithm']
        values = [
            row['Best Fitness'] if 'Best Fitness' in row else 0,
            row['Convergence Speed (iter)'] if 'Convergence Speed (iter)' in row else 0,
            row['Mean Time (s)'] if 'Mean Time (s)' in row else 0,
            row['Robustness'] if 'Robustness' in row else 0
        ]
        values += values[:1]
        
        color = colors.get(alg_name, 'gray')
        ax.plot(angles, values, 'o-', linewidth=2.5, label=alg_name,
               color=color, alpha=0.7)
        ax.fill(angles, values, alpha=0.15, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), framealpha=0.9, fontsize=11)
    
    plt.title(f'Algorithm Performance Radar - {problem_name.title()} Function',
             size=16, fontweight='bold', y=1.08)
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_path / f'{problem_name.lower()}_radar.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f" Saved: {output_file}")
    plt.close()


def plot_statistical_summary(problems=['sphere', 'ackley', 'rastrigin'],
                             results_dir='results/continuous/performance',
                             output_dir='visualizations/continuous/performance/overview'):
    """
    Create statistical summary heatmap.
    
    Args:
        problems: List of problem names
        results_dir: Directory containing results
        output_dir: Directory to save visualization
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, len(problems), figsize=(8 * len(problems), 6))
    
    if len(problems) == 1:
        axes = [axes]
    
    for idx, problem_name in enumerate(problems):
        df = load_metrics(problem_name, results_dir)
        
        if df is None:
            continue
        
        ax = axes[idx]
        
        # Prepare data for heatmap
        metrics = ['Mean Fitness', 'Convergence Speed (iter)', 'Mean Time (s)', 'Robustness']
        heatmap_data = df[['Algorithm'] + metrics].set_index('Algorithm')
        
        # Normalize each column to [0, 1]
        for col in metrics:
            if col != 'Robustness':
                # Lower is better
                max_val = heatmap_data[col].max()
                min_val = heatmap_data[col].min()
                if max_val > min_val:
                    heatmap_data[col] = 1 - (heatmap_data[col] - min_val) / (max_val - min_val)
        
        # Create heatmap
        sns.heatmap(heatmap_data.T, annot=True, fmt='.3f', cmap='RdYlGn',
                   center=0.5, linewidths=1, linecolor='white',
                   cbar_kws={'label': 'Normalized Score'}, ax=ax,
                   vmin=0, vmax=1)
        
        ax.set_title(f'{problem_name.title()} Function', fontsize=14, fontweight='bold')
        ax.set_xlabel('Algorithm', fontweight='bold')
        ax.set_ylabel('Metric', fontweight='bold')
    
    plt.suptitle('Statistical Performance Summary (Higher is Better)',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    output_file = output_path / 'statistical_summary.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def visualize_all_comparative(problems=['sphere', 'ackley', 'rastrigin'],
                              results_dir='results/continuous/performance',
                              output_dir='visualizations/continuous/performance/overview'):
    """
    Create all comparative visualizations.
    
    Args:
        problems: List of problem names
        results_dir: Directory containing results
        output_dir: Directory to save visualizations
    """
    print("""
=======================VISUALIZING COMPARATIVE PERFORMANCE=========================
    """)
    
    # Check if results exist
    missing_results = []
    for problem_name in problems:
        csv_file = Path(results_dir) / f'{problem_name.lower()}_metrics.csv'
        if not csv_file.exists():
            missing_results.append(problem_name)
    
    if missing_results:
        print(f"\nMissing results for: {', '.join(missing_results)}")
        print(f"Please run: python -m src.utils.compute_performance_metrics\n")
        return
    
    print("\nCreating performance comparison charts...")
    plot_performance_comparison(problems, results_dir, output_dir)
    
    print("\nCreating scalability comparison...")
    plot_scalability_comparison(problems, results_dir, output_dir)
    
    print("\nCreating radar charts...")
    for problem_name in problems:
        plot_radar_chart(problem_name, results_dir, output_dir)
    
    print("\nCreating statistical summary...")
    plot_statistical_summary(problems, results_dir, output_dir)
    
    print(f"""
======================VISUALIZATION COMPLETE!==========================

Visualizations saved to: {output_dir}
""")