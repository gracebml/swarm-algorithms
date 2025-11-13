"""
This module creates visualizations showing how algorithms converge
over iterations, comparing their convergence behavior.

Results are saved to visualizations/convergence/
"""

import numpy as np
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


def load_detailed_results(problem_name, results_dir='results/continuous/performance'):
    """
    Load detailed results from JSON file.
    
    Args:
        problem_name: Name of problem (sphere, ackley, rastrigin)
        results_dir: Directory containing results
    
    Returns:
        Dictionary with detailed results
    """
    json_file = Path(results_dir) / f'{problem_name.lower()}_detailed.json'
    
    if not json_file.exists():
        print(f" Results file not found: {json_file}")
        print(f"Please run compute_performance_metrics.py first")
        return None
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    return data


def plot_convergence_curves(data, problem_name, output_dir='visualizations/continuous/convergence'):
    """
    Plot convergence curves for all algorithms.
    
    Args:
        data: Dictionary with detailed results
        problem_name: Name of problem
        output_dir: Directory to save visualization
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Color palette
    colors = {
        'GA': '#2E86AB',
        'FA': '#A23B72',
        'PSO': '#F18F01',
        'SA': '#C73E1D',
        'CS': '#6A994E'
    }
    
    for alg_name, alg_data in data.items():
        convergence_curves = alg_data['convergence_curves']
        
        # Compute mean and std of convergence curves
        # Pad curves to same length
        max_len = max(len(curve) for curve in convergence_curves)
        padded_curves = []
        for curve in convergence_curves:
            if len(curve) < max_len:
                # Pad with last value
                padded = np.pad(curve, (0, max_len - len(curve)), 
                               mode='edge')
            else:
                padded = np.array(curve)
            padded_curves.append(padded)
        
        curves_array = np.array(padded_curves)
        mean_curve = np.mean(curves_array, axis=0)
        std_curve = np.std(curves_array, axis=0)
        
        iterations = np.arange(len(mean_curve))
        
        # Plot mean curve
        color = colors.get(alg_name, 'gray')
        ax.plot(iterations, mean_curve, linewidth=2.5, 
               label=alg_name, color=color, alpha=0.9)
        
        # Plot confidence interval
        ax.fill_between(iterations, 
                        mean_curve - std_curve, 
                        mean_curve + std_curve,
                        alpha=0.2, color=color)
    
    ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')
    ax.set_ylabel('Best Fitness (log scale)', fontsize=14, fontweight='bold')
    ax.set_title(f'Convergence Curves Comparison - {problem_name.title()} Function', 
                fontsize=16, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_path / f'{problem_name.lower()}_convergence.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f" Saved: {output_file}")
    plt.close()

def plot_convergence_comparison_grid(problems=['sphere', 'ackley', 'rastrigin'],
                                     results_dir='results/continuous/performance',
                                     output_dir='visualizations/continuous/convergence'):
    """
    Create a grid comparing convergence across multiple problems.
    
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
        data = load_detailed_results(problem_name, results_dir)
        
        if data is None:
            continue
        
        ax = axes[idx]
        
        for alg_name, alg_data in data.items():
            convergence_curves = alg_data['convergence_curves']
            
            # Compute mean curve
            max_len = max(len(curve) for curve in convergence_curves)
            padded_curves = []
            for curve in convergence_curves:
                if len(curve) < max_len:
                    padded = np.pad(curve, (0, max_len - len(curve)), mode='edge')
                else:
                    padded = np.array(curve)
                padded_curves.append(padded)
            
            curves_array = np.array(padded_curves)
            mean_curve = np.mean(curves_array, axis=0)
            
            iterations = np.arange(len(mean_curve))
            color = colors.get(alg_name, 'gray')
            ax.plot(iterations, mean_curve, linewidth=2.5,
                   label=alg_name, color=color, alpha=0.9)
        
        ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Best Fitness', fontsize=12, fontweight='bold')
        ax.set_title(f'{problem_name.title()} Function', fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.suptitle('Convergence Comparison Across Problems',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    output_file = output_path / 'convergence_comparison_grid.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f" Saved: {output_file}")
    plt.close()


def visualize_all_convergence(problems=['sphere', 'ackley', 'rastrigin'],
                              results_dir='results/continuous/performance',
                              output_dir='visualizations/continuous/convergence'):
    """
    Create all convergence visualizations.
    
    Args:
        problems: List of problem names
        results_dir: Directory containing results
        output_dir: Directory to save visualizations
    """
    print("""
==================VISUALIZING CONVERGENCE ABILITY=========================
    """)
    
    # Check if results exist
    missing_results = []
    for problem_name in problems:
        json_file = Path(results_dir) / f'{problem_name.lower()}_detailed.json'
        if not json_file.exists():
            missing_results.append(problem_name)
    
    if missing_results:
        print(f"\n Missing results for: {', '.join(missing_results)}")
        print(f"Please run: python main.py (Custom Mode > Performance Metrics Computation)\n")
        return
    
    # Create individual convergence plots
    for problem_name in problems:
        print(f"\nVisualizing {problem_name.title()} Function:")
        data = load_detailed_results(problem_name, results_dir)
        
        if data:
            plot_convergence_curves(data, problem_name, output_dir)
            # plot_convergence_rate(data, problem_name, output_dir)
    
    # Create comparison grid
    print(f"\nCreating comparison grid...")
    plot_convergence_comparison_grid(problems, results_dir, output_dir)
    
    print(f"""
======================VISUALIZATION COMPLETE!==========================


Visualizations saved to: {output_dir}
    """)



