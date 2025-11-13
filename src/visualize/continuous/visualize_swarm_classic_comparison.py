"""
This module compares:
- Swarm-based: PSO, FA, CS, ABC (population-based, social learning)
- Classical: GA, SA (evolution-based, local search)

Highlights paradigm differences, strengths/weaknesses, and when to use each approach.
Results are saved to visualizations/performance/swarm_classic_comparison/
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

# Define paradigm groups
SWARM_ALGORITHMS = ['PSO', 'FA', 'CS', 'ABC']
CLASSICAL_ALGORITHMS = ['GA', 'SA']

# Color schemes
PARADIGM_COLORS = {
    'Swarm-based': '#F18F01',  # Orange
    'Classical': '#2E86AB'      # Blue
}

ALGORITHM_COLORS = {
    # Swarm-based (warm colors)
    'PSO': '#F18F01',
    'FA': '#A23B72',
    'CS': '#6A994E',
    'ABC': '#596A4E',
    # Classical (cool colors)
    'GA': '#2E86AB',
    'SA': '#C73E1D'
}


def load_metrics(problem_name, results_dir='results/continuous/performance'):
    """Load performance metrics from CSV file."""
    csv_file = Path(results_dir) / f'{problem_name.lower()}_metrics.csv'
    
    if not csv_file.exists():
        print(f" Metrics file not found: {csv_file}")
        return None
    
    return pd.read_csv(csv_file)


def load_scalability_data(problem_name, results_dir='results/continuous/performance'):
    """Load scalability data from JSON file."""
    json_file = Path(results_dir) / f'{problem_name.lower()}_scalability_data.json'
    
    if not json_file.exists():
        print(f" Scalability file not found: {json_file}")
        return None
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    return data


def classify_algorithm(alg_name):
    """Classify algorithm as Swarm or Classical."""
    if alg_name in SWARM_ALGORITHMS:
        return 'Swarm-based'
    elif alg_name in CLASSICAL_ALGORITHMS:
        return 'Classical'
    return 'Unknown'


def plot_side_by_side_comparison(problems=['sphere', 'ackley', 'rastrigin'],
                                  results_dir='results/continuous/performance',
                                  output_dir='visualizations/continuous/performance/swarm_classic_comparison'):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Collect data
    all_data = []
    for problem_name in problems:
        df = load_metrics(problem_name, results_dir)
        if df is not None:
            df['Problem'] = problem_name.title()
            df['Paradigm'] = df['Algorithm'].apply(classify_algorithm)
            all_data.append(df)
    
    if not all_data:
        print(" No data available")
        return
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Filter to only include known algorithms
    combined_df = combined_df[combined_df['Paradigm'] != 'Unknown']
    
    # Create figure with 4 metrics
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    metrics = [
        ('Best Fitness', 'Best Fitness Value (log scale)', True),
        ('Convergence Speed (iter)', 'Iterations to Converge', False),
        ('Mean Time (s)', 'Execution Time (seconds)', False),
        ('Robustness', 'Robustness Score', False)
    ]
    
    for idx, (metric, ylabel, log_scale) in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        # Separate swarm and classical
        swarm_df = combined_df[combined_df['Paradigm'] == 'Swarm-based']
        classical_df = combined_df[combined_df['Paradigm'] == 'Classical']
        
        # Get unique problems for x-axis
        problems_list = combined_df['Problem'].unique()
        x = np.arange(len(problems_list))
        width = 0.12  # Width of bars
        
        # Plot swarm algorithms
        for i, alg in enumerate(SWARM_ALGORITHMS):
            alg_data = swarm_df[swarm_df['Algorithm'] == alg]
            if not alg_data.empty:
                values = [alg_data[alg_data['Problem'] == p][metric].values[0] 
                         if len(alg_data[alg_data['Problem'] == p]) > 0 else 0 
                         for p in problems_list]
                offset = (i - len(SWARM_ALGORITHMS)/2 + 0.5) * width - 0.3
                ax.bar(x + offset, values, width, label=alg,
                      color=ALGORITHM_COLORS.get(alg, 'gray'), alpha=0.8)
        
        # Plot classical algorithms
        for i, alg in enumerate(CLASSICAL_ALGORITHMS):
            alg_data = classical_df[classical_df['Algorithm'] == alg]
            if not alg_data.empty:
                values = [alg_data[alg_data['Problem'] == p][metric].values[0] 
                         if len(alg_data[alg_data['Problem'] == p]) > 0 else 0 
                         for p in problems_list]
                offset = (i - len(CLASSICAL_ALGORITHMS)/2 + 0.5) * width + 0.3
                ax.bar(x + offset, values, width, label=alg,
                      color=ALGORITHM_COLORS.get(alg, 'gray'), alpha=0.8)
        
        # Add vertical separator line between paradigms
        ax.axvline(x=-0.05, color='gray', linestyle='--', linewidth=2, alpha=0.5)
        
        # Add paradigm labels
        ax.text(-0.35, ax.get_ylim()[1] * 0.95, 'Swarm', 
               fontsize=11, fontweight='bold', rotation=90, va='top',
               color=PARADIGM_COLORS['Swarm-based'])
        ax.text(0.35, ax.get_ylim()[1] * 0.95, 'Classical', 
               fontsize=11, fontweight='bold', rotation=90, va='top',
               color=PARADIGM_COLORS['Classical'])
        
        ax.set_ylabel(ylabel, fontweight='bold')
        ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
        if log_scale:
            ax.set_yscale('log')
        ax.set_xticks(x)
        ax.set_xticklabels(problems_list, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(loc='upper left', framealpha=0.9, fontsize=10, ncol=2)
    
    plt.suptitle('Paradigm Comparison: Swarm-based vs Classical Algorithms',
                fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save figure
    output_file = output_path / 'side_by_side_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f" Saved: {output_file}")
    plt.close()


def plot_category_averages(problems=['sphere', 'ackley', 'rastrigin'],
                           results_dir='results/continuous/performance',
                           output_dir='visualizations/performance/swarm_vs_classical_comparison'):
    """
    Plot category averages comparing Swarm-based vs Classical paradigms.
    
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
            df['Paradigm'] = df['Algorithm'].apply(classify_algorithm)
            all_data.append(df)
    
    if not all_data:
        print(" No data available")
        return
    
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df[combined_df['Paradigm'] != 'Unknown']
    
    # Calculate category averages
    metrics = ['Best Fitness', 'Convergence Speed (iter)', 'Mean Time (s)', 'Robustness']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        
        # Calculate averages by paradigm and problem
        avg_data = combined_df.groupby(['Paradigm', 'Problem'])[metric].mean().reset_index()
        
        # Pivot for grouped bar chart
        pivot_df = avg_data.pivot(index='Problem', columns='Paradigm', values=metric)
        
        # Reorder columns to ensure consistent order
        if 'Swarm-based' in pivot_df.columns and 'Classical' in pivot_df.columns:
            pivot_df = pivot_df[['Swarm-based', 'Classical']]
        
        # Plot
        x = np.arange(len(pivot_df.index))
        width = 0.35
        
        colors_list = [PARADIGM_COLORS['Swarm-based'], PARADIGM_COLORS['Classical']]
        pivot_df.plot(kind='bar', ax=ax, color=colors_list, alpha=0.85, width=0.7)
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f' if metric == 'Robustness' else '%.2f',
                        padding=3, fontsize=10, fontweight='bold')
        
        ax.set_ylabel(metric, fontweight='bold', fontsize=13)
        ax.set_title(f'Average {metric}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Problem', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(title='Paradigm', framealpha=0.9, fontsize=11)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')
    
    plt.suptitle('Category Averages: Swarm-based vs Classical',
                fontsize=18, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save figure
    output_file = output_path / 'category_averages.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f" Saved: {output_file}")
    plt.close()


def plot_paradigm_strengths_heatmap(problems=['sphere', 'ackley', 'rastrigin'],
                                    results_dir='results/continuous/performance',
                                    output_dir='visualizations/continuous/performance/swarm_classic_comparison'):
    """
    Create heatmap showing strengths of each paradigm across metrics.
    
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
            df['Paradigm'] = df['Algorithm'].apply(classify_algorithm)
            all_data.append(df)
    
    if not all_data:
        print(" No data available")
        return
    
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df[combined_df['Paradigm'] != 'Unknown']
    
    # Calculate normalized scores for each paradigm
    metrics = ['Best Fitness', 'Convergence Speed (iter)', 'Mean Time (s)', 'Robustness']
    metric_labels = ['Solution Quality', 'Convergence Speed', 'Time Efficiency', 'Robustness']
    
    # Group by paradigm and problem, calculate mean
    paradigm_scores = []
    
    for problem in combined_df['Problem'].unique():
        problem_df = combined_df[combined_df['Problem'] == problem]
        
        for paradigm in ['Swarm-based', 'Classical']:
            paradigm_df = problem_df[problem_df['Paradigm'] == paradigm]
            
            if paradigm_df.empty:
                continue
            
            scores = {}
            for metric, label in zip(metrics, metric_labels):
                # Normalize: for most metrics, lower is better (except Robustness)
                all_values = problem_df[metric].values
                paradigm_value = paradigm_df[metric].mean()
                
                if metric == 'Robustness':
                    # Higher is better - normalize to [0, 1]
                    score = paradigm_value
                else:
                    # Lower is better - invert and normalize
                    max_val = all_values.max()
                    min_val = all_values.min()
                    if max_val > min_val:
                        score = 1 - (paradigm_value - min_val) / (max_val - min_val)
                    else:
                        score = 1.0
                
                scores[label] = score
            
            paradigm_scores.append({
                'Problem': problem,
                'Paradigm': paradigm,
                **scores
            })
    
    # Create DataFrame
    scores_df = pd.DataFrame(paradigm_scores)
    
    # Create heatmap for each problem
    fig, axes = plt.subplots(1, len(problems), figsize=(7 * len(problems), 6))
    if len(problems) == 1:
        axes = [axes]
    
    for idx, problem in enumerate(combined_df['Problem'].unique()):
        ax = axes[idx]
        
        problem_data = scores_df[scores_df['Problem'] == problem]
        heatmap_data = problem_data[metric_labels].set_index(problem_data['Paradigm'])
        
        # Create heatmap
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn',
                   center=0.5, linewidths=2, linecolor='white',
                   cbar_kws={'label': 'Normalized Score'},
                   vmin=0, vmax=1, ax=ax,
                   annot_kws={'fontsize': 12, 'fontweight': 'bold'})
        
        ax.set_title(f'{problem} Function', fontsize=14, fontweight='bold')
        ax.set_xlabel('Metric', fontweight='bold')
        ax.set_ylabel('Paradigm', fontweight='bold')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    plt.suptitle('Paradigm Strengths Analysis (Higher is Better)',
                fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    output_file = output_path / 'paradigm_strengths_heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f" Saved: {output_file}")
    plt.close()


def plot_performance_profiles(problems=['sphere', 'ackley', 'rastrigin'],
                              results_dir='results/continuous/performance',
                              output_dir='visualizations/continuous/performance/swarm_classic_comparison'):
    """
    Create radar chart comparing paradigm performance profiles.
    
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
            df['Paradigm'] = df['Algorithm'].apply(classify_algorithm)
            all_data.append(df)
    
    if not all_data:
        print(" No data available")
        return
    
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df[combined_df['Paradigm'] != 'Unknown']
    
    # Create subplots for each problem
    fig, axes = plt.subplots(1, len(problems), figsize=(8 * len(problems), 8),
                            subplot_kw=dict(polar=True))
    if len(problems) == 1:
        axes = [axes]
    
    categories = ['Quality', 'Speed', 'Time\nEfficiency', 'Robustness']
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    
    for idx, problem in enumerate(combined_df['Problem'].unique()):
        ax = axes[idx]
        problem_df = combined_df[combined_df['Problem'] == problem]
        
        for paradigm in ['Swarm-based', 'Classical']:
            paradigm_df = problem_df[problem_df['Paradigm'] == paradigm]
            
            if paradigm_df.empty:
                continue
            
            # Calculate normalized average scores
            metrics = ['Best Fitness', 'Convergence Speed (iter)', 'Mean Time (s)', 'Robustness']
            values = []
            
            for metric in metrics:
                all_values = problem_df[metric].values
                paradigm_value = paradigm_df[metric].mean()
                
                if metric == 'Robustness':
                    score = paradigm_value
                else:
                    max_val = all_values.max()
                    min_val = all_values.min()
                    if max_val > min_val:
                        score = 1 - (paradigm_value - min_val) / (max_val - min_val)
                    else:
                        score = 1.0
                
                values.append(score)
            
            values += values[:1]
            
            color = PARADIGM_COLORS[paradigm]
            ax.plot(angles, values, 'o-', linewidth=3, label=paradigm,
                   color=color, alpha=0.8, markersize=8)
            ax.fill(angles, values, alpha=0.2, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=13, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), 
                 framealpha=0.9, fontsize=12)
        ax.set_title(f'{problem} Function', size=14, fontweight='bold', y=1.08)
    
    plt.suptitle('Paradigm Performance Profiles',
                fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # Save figure
    output_file = output_path / 'paradigm_performance_profiles.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f" Saved: {output_file}")
    plt.close()


def plot_scalability_paradigm_comparison(problems=['sphere', 'ackley', 'rastrigin'],
                                        results_dir='results/continuous/performance',
                                        output_dir='visualizations/continuous/performance/swarm_classic_comparison'):
    """
    Compare scalability between Swarm and Classical paradigms.
    
    Args:
        problems: List of problem names
        results_dir: Directory containing results
        output_dir: Directory to save visualization
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subplots
    fig, axes = plt.subplots(1, len(problems), figsize=(7 * len(problems), 6))
    if len(problems) == 1:
        axes = [axes]
    
    for idx, problem_name in enumerate(problems):
        data = load_scalability_data(problem_name, results_dir)
        
        if data is None:
            continue
        
        ax = axes[idx]
        
        # Separate by paradigm
        swarm_data = {k: v for k, v in data.items() if k in SWARM_ALGORITHMS}
        classical_data = {k: v for k, v in data.items() if k in CLASSICAL_ALGORITHMS}
        
        # Plot swarm algorithms with thinner lines
        for alg_name, alg_data in swarm_data.items():
            dimensions = alg_data['dimensions']
            times = alg_data['times']
            color = ALGORITHM_COLORS.get(alg_name, 'gray')
            ax.plot(dimensions, times, marker='o', linewidth=2,
                   markersize=7, label=f'{alg_name} (Swarm)', 
                   color=color, alpha=0.7, linestyle='-')
        
        # Plot classical algorithms with thicker lines
        for alg_name, alg_data in classical_data.items():
            dimensions = alg_data['dimensions']
            times = alg_data['times']
            color = ALGORITHM_COLORS.get(alg_name, 'gray')
            ax.plot(dimensions, times, marker='s', linewidth=3,
                   markersize=8, label=f'{alg_name} (Classical)', 
                   color=color, alpha=0.8, linestyle='--')
        
        ax.set_xlabel('Problem Dimensions', fontweight='bold', fontsize=13)
        ax.set_ylabel('Execution Time (seconds)', fontweight='bold', fontsize=13)
        ax.set_title(f'{problem_name.title()} Function', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', framealpha=0.9, fontsize=10)
    
    plt.suptitle('Scalability: Swarm-based vs Classical\n(Flatter = Better Scalability)',
                fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    output_file = output_path / 'scalability_paradigm_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f" Saved: {output_file}")
    plt.close()

def visualize_all_paradigm_comparison(problems=['sphere', 'ackley', 'rastrigin'],
                                      results_dir='results/continuous/performance',
                                      output_dir='visualizations/continuous/performance/swarm_classic_comparison'):
    print("""
===================PARADIGM COMPARISON: SWARM-BASED VS CLASSICAL====================
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
    
    print("\n1. Creating side-by-side comparison...")
    plot_side_by_side_comparison(problems, results_dir, output_dir)
    
    print("\n2. Creating category averages...")
    plot_category_averages(problems, results_dir, output_dir)
    
    print("\n3. Creating paradigm strengths heatmap...")
    plot_paradigm_strengths_heatmap(problems, results_dir, output_dir)
    
    print("\n4. Creating performance profiles...")
    plot_performance_profiles(problems, results_dir, output_dir)
    
    print("\n5. Creating scalability comparison...")
    plot_scalability_paradigm_comparison(problems, results_dir, output_dir)
    
    print(f"""
===================VISUALIZATION COMPLETE!============================
Visualizations saved to: {output_dir}

Generated files:
  1. side_by_side_comparison.png       - Direct comparison of all algorithms
  2. category_averages.png             - Average performance by paradigm
  3. paradigm_strengths_heatmap.png    - Normalized strength comparison
  4. paradigm_performance_profiles.png - Radar charts by problem
  5. scalability_paradigm_comparison.png - Scalability analysis

Key Insights:
  - Swarm-based: {', '.join(SWARM_ALGORITHMS)}
  - Classical: {', '.join(CLASSICAL_ALGORITHMS)}
  
Use these visualizations to understand when to choose Swarm vs Classical!
    """)