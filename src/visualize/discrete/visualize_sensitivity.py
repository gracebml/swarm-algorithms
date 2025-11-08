"""
Visualize parameter sensitivity analysis results for discrete algorithms (ACO).
"""

import os
import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional

# Setup path
current_file = Path(__file__).resolve()
src_dir = current_file.parent.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


def load_sensitivity_data(results_dir: str, algorithm: str = 'aco') -> Dict:
    """Load sensitivity analysis results for an algorithm."""
    alg_dir = Path(results_dir) / algorithm
    
    if not alg_dir.exists():
        return {}
    
    data = {}
    
    # Load all sensitivity files
    for file_path in alg_dir.glob('*_sensitivity.csv'):
        param_name = file_path.stem.replace(f'{algorithm}_', '').replace('_sensitivity', '')
        
        # Read CSV data
        param_values = []
        mean_fitness = []
        std_fitness = []
        
        with open(file_path, 'r') as f:
            lines = f.readlines()[1:]  # Skip header
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    param_values.append(float(parts[0]))
                    mean_fitness.append(float(parts[1]))
                    std_fitness.append(float(parts[2]))
        
        data[param_name] = {
            'param_values': param_values,
            'mean_fitness': mean_fitness,
            'std_fitness': std_fitness
        }
    
    return data


def plot_parameter_sensitivity(
    data: Dict,
    algorithm: str,
    output_dir: str
):
    """Plot parameter sensitivity for each parameter."""
    
    output_path = Path(output_dir) / algorithm
    output_path.mkdir(parents=True, exist_ok=True)
    
    for param_name, param_data in data.items():
        param_values = param_data['param_values']
        mean_fitness = param_data['mean_fitness']
        std_fitness = param_data['std_fitness']
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.errorbar(param_values, mean_fitness, yerr=std_fitness,
                    marker='o', linewidth=2, capsize=5, capthick=2)
        plt.xlabel(param_name.replace('_', ' ').title(), fontsize=12)
        plt.ylabel('Tour Length (Mean ± Std)', fontsize=12)
        plt.title(f'{algorithm.upper()} - {param_name.replace("_", " ").title()} Sensitivity',
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Save
        save_path = output_path / f'{param_name}_sensitivity.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {save_path}")


def plot_parameter_heatmap(
    data: Dict,
    algorithm: str,
    output_dir: str
):
    """Plot heatmap of parameter sensitivities."""
    
    if not data:
        return
    
    output_path = Path(output_dir) / algorithm
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare data for heatmap
    params = list(data.keys())
    
    # Normalize fitness values and compute sensitivity scores
    sensitivity_matrix = []
    
    for param_name in params:
        mean_fitness = np.array(data[param_name]['mean_fitness'])
        # Compute coefficient of variation as sensitivity measure
        cv = (np.std(mean_fitness) / np.mean(mean_fitness)) * 100
        sensitivity_matrix.append([cv])
    
    # Create heatmap
    plt.figure(figsize=(8, len(params) * 0.6))
    sns.heatmap([s for s in sensitivity_matrix],
                annot=True,
                fmt='.2f',
                cmap='YlOrRd',
                yticklabels=[p.replace('_', ' ').title() for p in params],
                xticklabels=['Sensitivity (CV %)'],
                cbar_kws={'label': 'Coefficient of Variation (%)'})
    
    plt.title(f'{algorithm.upper()} - Parameter Sensitivity Overview',
             fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = output_path / 'sensitivity_heatmap.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {save_path}")


def visualize_algorithm_results(
    algorithm_name: str,
    input_dir: str = 'results/discrete/parameter_sensitivity',
    output_dir: str = 'visualizations/discrete/parameter_sensitivity',
    auto_run: bool = True
):    
    print(f"\nVisualizing sensitivity results for {algorithm_name.upper()}...")
    
    # Check if results exist
    alg_dir = Path(input_dir) / algorithm_name
    if not alg_dir.exists() or not list(alg_dir.glob('*_sensitivity.csv')):
        if auto_run:
            print(f"  No results found. Auto-running sensitivity analysis...")
            try:
                from utils.run_sensitivity_discrete import run_sensitivity_analysis
                config_file = f'configs/sensitivity/{algorithm_name}_sensitivity_config.yaml'
                run_sensitivity_analysis(config_file)
            except Exception as e:
                print(f"  Error running sensitivity analysis: {e}")
                return
        else:
            print(f"  No results found in {alg_dir}")
            return
    
    # Load data
    data = load_sensitivity_data(input_dir, algorithm_name)
    
    if not data:
        print(f"  No data loaded for {algorithm_name}")
        return
    
    # Create visualizations
    print(f"  Creating visualizations...")
    plot_parameter_sensitivity(data, algorithm_name, output_dir)
    plot_parameter_heatmap(data, algorithm_name, output_dir)
    
    print(f"  Completed visualization for {algorithm_name}!")


def visualize_all_results(
    input_dir: str = 'results/discrete/parameter_sensitivity',
    output_dir: str = 'visualizations/discrete/parameter_sensitivity',
    auto_run: bool = True
):
    """Visualize sensitivity analysis results for all discrete algorithms."""
    
    algorithms = ['aco']  # Add more discrete algorithms here if needed
    
    print("\n" + "="*80)
    print(" Visualizing Sensitivity Analysis - Discrete Algorithms")
    print("="*80)
    
    for algorithm in algorithms:
        visualize_algorithm_results(
            algorithm,
            input_dir=input_dir,
            output_dir=output_dir,
            auto_run=auto_run
        )
    
    print("\n" + "="*80)
    print(" All visualizations completed!")
    print(f" Output directory: {output_dir}")
    print("="*80 + "\n")

