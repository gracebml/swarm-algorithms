"""
Visualize individual metrics for swarm-based algorithms only.

Focus on swarm-based algorithms: PSO, FA, CS, ABC
Results are saved to visualizations/performance/swarm_only/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# Setup matplotlib style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12

# Define swarm algorithms
SWARM_ALGORITHMS = ['PSO', 'FA', 'CS', 'ABC']
SWARM_COLORS = {
    'PSO': '#F18F01',
    'FA': '#A23B72',
    'CS': '#6A994E',
    'ABC': '#596A4E'
}


def load_metrics(problem_name, results_dir='results/continuous/performance'):
    """Load performance metrics from CSV file."""
    csv_file = Path(results_dir) / f'{problem_name.lower()}_metrics.csv'
    
    if not csv_file.exists():
        print(f" Metrics file not found: {csv_file}")
        return None
    
    df = pd.read_csv(csv_file)
    # Filter only swarm algorithms
    df = df[df['Algorithm'].isin(SWARM_ALGORITHMS)]
    return df


def load_scalability_data(problem_name, results_dir='results/continuous/performance'):
    """Load scalability data from JSON file."""
    json_file = Path(results_dir) / f'{problem_name.lower()}_scalability_data.json'
    
    if not json_file.exists():
        print(f" Scalability file not found: {json_file}")
        return None
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Filter only swarm algorithms
    swarm_data = {k: v for k, v in data.items() if k in SWARM_ALGORITHMS}
    return swarm_data


def plot_convergence_speed_metric(problems=['sphere', 'ackley', 'rastrigin'],
                                   results_dir='results/continuous/performance',
                                   output_dir='visualizations/continuous/performance/swarm_only'):
    """
    Visualize Convergence Speed metric for swarm algorithms.
    
    Args:
        problems: List of problem names
        results_dir: Directory containing results
        output_dir: Directory to save visualization
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Collect data
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
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Prepare data for grouped bar chart
    pivot_df = combined_df.pivot(index='Algorithm', columns='Problem', 
                                  values='Convergence Speed (iter)')
    
    # Plot grouped bars
    x = np.arange(len(pivot_df.index))
    width = 0.25
    
    for idx, problem in enumerate(pivot_df.columns):
        offset = (idx - len(pivot_df.columns)/2 + 0.5) * width
        bars = ax.bar(x + offset, pivot_df[problem], width, 
                     label=problem, alpha=0.8)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Swarm Algorithm', fontsize=14, fontweight='bold')
    ax.set_ylabel('Iterations to Converge', fontsize=14, fontweight='bold')
    ax.set_title('Convergence Speed Comparison\n(Lower is Better)', 
                fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_df.index, fontsize=12, fontweight='bold')
    ax.legend(title='Problem', framealpha=0.9, fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_path / 'convergence_speed_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f" Saved: {output_file}")
    plt.close()
    
    # Create individual problem plots
    fig, axes = plt.subplots(1, len(problems), figsize=(5*len(problems), 6))
    if len(problems) == 1:
        axes = [axes]
    
    for idx, problem_name in enumerate(problems):
        df = combined_df[combined_df['Problem'] == problem_name.title()]
        ax = axes[idx]
        
        bars = ax.bar(df['Algorithm'], df['Convergence Speed (iter)'],
                     color=[SWARM_COLORS.get(alg, 'gray') for alg in df['Algorithm']],
                     alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel('Iterations to Converge', fontsize=12, fontweight='bold')
        ax.set_title(f'{problem_name.title()} Function', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        # Set ticks explicitly before setting labels to avoid warning
        ax.set_xticks(range(len(df['Algorithm'])))
        ax.set_xticklabels(df['Algorithm'], fontsize=11, fontweight='bold')
    
    plt.suptitle('Convergence Speed by Problem (Lower is Better)',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_file = output_path / 'convergence_speed_by_problem.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f" Saved: {output_file}")
    plt.close()


def plot_computational_time_metric(problems=['sphere', 'ackley', 'rastrigin'],
                                    results_dir='results/continuous/performance',
                                    output_dir='visualizations/continuous/performance/swarm_only'):
    """
    Visualize Computational Time metric for swarm algorithms.
    
    Args:
        problems: List of problem names
        results_dir: Directory containing results
        output_dir: Directory to save visualization
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Collect data
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
    
    # Create figure with error bars
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Prepare data
    pivot_mean = combined_df.pivot(index='Algorithm', columns='Problem', 
                                    values='Mean Time (s)')
    pivot_std = combined_df.pivot(index='Algorithm', columns='Problem', 
                                   values='Std Time (s)')
    
    # Plot grouped bars with error bars
    x = np.arange(len(pivot_mean.index))
    width = 0.25
    
    for idx, problem in enumerate(pivot_mean.columns):
        offset = (idx - len(pivot_mean.columns)/2 + 0.5) * width
        bars = ax.bar(x + offset, pivot_mean[problem], width, 
                     yerr=pivot_std[problem], capsize=5,
                     label=problem, alpha=0.8, error_kw={'linewidth': 2})
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}s',
                       ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Swarm Algorithm', fontsize=14, fontweight='bold')
    ax.set_ylabel('Execution Time (seconds)', fontsize=14, fontweight='bold')
    ax.set_title('Computational Time Comparison\n(Lower is Better)', 
                fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_mean.index, fontsize=12, fontweight='bold')
    ax.legend(title='Problem', framealpha=0.9, fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_path / 'computational_time_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f" Saved: {output_file}")
    plt.close()
    
    # Create box plot for time distribution
    fig, axes = plt.subplots(1, len(problems), figsize=(5*len(problems), 6))
    if len(problems) == 1:
        axes = [axes]
    
    for idx, problem_name in enumerate(problems):
        df = combined_df[combined_df['Problem'] == problem_name.title()]
        ax = axes[idx]
        
        # Create data for box plot
        data_to_plot = []
        labels = []
        colors = []
        for _, row in df.iterrows():
            # Simulate distribution based on mean and std
            mean_time = row['Mean Time (s)']
            std_time = row['Std Time (s)']
            simulated_data = np.random.normal(mean_time, std_time, 30)
            data_to_plot.append(simulated_data)
            labels.append(row['Algorithm'])
            colors.append(SWARM_COLORS.get(row['Algorithm'], 'gray'))
        
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                        showmeans=True, meanline=True)
        
        # Color boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title(f'{problem_name.title()} Function', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Execution Time Distribution (Lower is Better)',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_file = output_path / 'computational_time_distribution.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f" Saved: {output_file}")
    plt.close()


def plot_robustness_metric(problems=['sphere', 'ackley', 'rastrigin'],
                           results_dir='results/continuous/performance',
                           output_dir='visualizations/continuous/performance/swarm_only'):
    """
    Visualize Robustness metric for swarm algorithms.
    
    Args:
        problems: List of problem names
        results_dir: Directory containing results
        output_dir: Directory to save visualization
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Collect data
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
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Prepare data
    pivot_df = combined_df.pivot(index='Algorithm', columns='Problem', 
                                  values='Robustness')
    
    # Plot grouped bars
    x = np.arange(len(pivot_df.index))
    width = 0.25
    
    for idx, problem in enumerate(pivot_df.columns):
        offset = (idx - len(pivot_df.columns)/2 + 0.5) * width
        bars = ax.bar(x + offset, pivot_df[problem], width, 
                     label=problem, alpha=0.8)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Swarm Algorithm', fontsize=14, fontweight='bold')
    ax.set_ylabel('Robustness Score', fontsize=14, fontweight='bold')
    ax.set_title('Robustness Comparison\n(Higher is Better, Max = 1.0)', 
                fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_df.index, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Perfect')
    ax.legend(title='Problem', framealpha=0.9, fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_path / 'robustness_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f" Saved: {output_file}")
    plt.close()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for heatmap
    heatmap_data = combined_df.pivot(index='Algorithm', columns='Problem', 
                                      values='Robustness')
    
    # Create heatmap
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn',
               center=0.5, linewidths=2, linecolor='white',
               cbar_kws={'label': 'Robustness Score'},
               vmin=0, vmax=1, ax=ax, annot_kws={'fontsize': 12, 'fontweight': 'bold'})
    
    ax.set_title('Robustness Heatmap (Higher is Better)',
                fontsize=16, fontweight='bold')
    ax.set_xlabel('Problem', fontsize=14, fontweight='bold')
    ax.set_ylabel('Swarm Algorithm', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    output_file = output_path / 'robustness_heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f" Saved: {output_file}")
    plt.close()


def plot_scalability_metric(problems=['sphere', 'ackley', 'rastrigin'],
                            results_dir='results/continuous/performance',
                            output_dir='visualizations/continuous/performance/swarm_only'):
    """
    Visualize Scalability metric for swarm algorithms.
    
    Args:
        problems: List of problem names
        results_dir: Directory containing results
        output_dir: Directory to save visualization
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subplots for each problem
    fig, axes = plt.subplots(1, len(problems), figsize=(6*len(problems), 6))
    if len(problems) == 1:
        axes = [axes]
    
    for idx, problem_name in enumerate(problems):
        data = load_scalability_data(problem_name, results_dir)
        
        if data is None:
            continue
        
        ax = axes[idx]
        
        for alg_name, alg_data in data.items():
            dimensions = alg_data['dimensions']
            times = alg_data['times']
            
            color = SWARM_COLORS.get(alg_name, 'gray')
            
            # Plot with markers
            ax.plot(dimensions, times, marker='o', linewidth=3,
                   markersize=10, label=alg_name, color=color, alpha=0.8)
            
            # Fit linear trend line
            z = np.polyfit(dimensions, times, 1)
            p = np.poly1d(z)
            ax.plot(dimensions, p(dimensions), linestyle='--', linewidth=1.5,
                   color=color, alpha=0.5, label=f'{alg_name} trend')
        
        ax.set_xlabel('Problem Dimensions', fontsize=12, fontweight='bold')
        ax.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title(f'{problem_name.title()} Function', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', framealpha=0.9, fontsize=10)
    
    plt.suptitle('Scalability Analysis\n(Flatter Curve = Better Scalability)',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    output_file = output_path / 'scalability_curves.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f" Saved: {output_file}")
    plt.close()
    
    # Create scalability coefficient comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    coefficients = []
    for problem_name in problems:
        data = load_scalability_data(problem_name, results_dir)
        if data:
            for alg_name, alg_data in data.items():
                dimensions = alg_data['dimensions']
                times = alg_data['times']
                
                # Compute slope (scalability coefficient)
                coeffs = np.polyfit(dimensions, times, 1)
                slope = coeffs[0]
                
                coefficients.append({
                    'Algorithm': alg_name,
                    'Problem': problem_name.title(),
                    'Coefficient': slope
                })
    
    if coefficients:
        coeff_df = pd.DataFrame(coefficients)
        pivot_df = coeff_df.pivot(index='Algorithm', columns='Problem', 
                                   values='Coefficient')
        
        # Plot
        x = np.arange(len(pivot_df.index))
        width = 0.25
        
        for idx, problem in enumerate(pivot_df.columns):
            offset = (idx - len(pivot_df.columns)/2 + 0.5) * width
            bars = ax.bar(x + offset, pivot_df[problem], width, 
                         label=problem, alpha=0.8)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.4f}',
                           ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Swarm Algorithm', fontsize=14, fontweight='bold')
        ax.set_ylabel('Scalability Coefficient (slope)', fontsize=14, fontweight='bold')
        ax.set_title('Scalability Coefficient Comparison\n(Lower = Better Scalability)', 
                    fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(pivot_df.index, fontsize=12, fontweight='bold')
        ax.legend(title='Problem', framealpha=0.9, fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        output_file = output_path / 'scalability_coefficients.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f" Saved: {output_file}")
        plt.close()


def visualize_all_swarm_metrics(problems=['sphere', 'ackley', 'rastrigin'],
                                results_dir='results/continuous/performance',
                                output_dir='visualizations/continuous/performance/swarm_only'):
    """
    Create all individual metric visualizations for swarm algorithms.
    
    Args:
        problems: List of problem names
        results_dir: Directory containing results
        output_dir: Directory to save visualizations
    """
    print("""
===================VISUALIZING SWARM ALGORITHM METRICS=========================
    """)
    
    # Check if results exist
    missing_results = []
    for problem_name in problems:
        csv_file = Path(results_dir) / f'{problem_name.lower()}_metrics.csv'
        if not csv_file.exists():
            missing_results.append(problem_name)
    
    if missing_results:
        print(f"\n Missing results for: {', '.join(missing_results)}")
        print(f"Please run: python main.py (Custom Mode > Performance Metrics Computation)\n")
        return
    
    print("\n1. Visualizing Convergence Speed metric...")
    plot_convergence_speed_metric(problems, results_dir, output_dir)
    
    print("\n2. Visualizing Computational Time metric...")
    plot_computational_time_metric(problems, results_dir, output_dir)
    
    print("\n3. Visualizing Robustness metric...")
    plot_robustness_metric(problems, results_dir, output_dir)
    
    print("\n4. Visualizing Scalability metric...")
    plot_scalability_metric(problems, results_dir, output_dir)
    
    print(f"""
=========================VISUALIZATION COMPLETE!============================

Visualizations saved to: {output_dir}

Generated files:
  - convergence_speed_comparison.png
  - convergence_speed_by_problem.png
  - computational_time_comparison.png
  - computational_time_distribution.png
  - robustness_comparison.png
  - robustness_heatmap.png
  - scalability_curves.png
  - scalability_coefficients.png
    """)
