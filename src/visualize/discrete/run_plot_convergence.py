import csv
import matplotlib.pyplot as plt
import os
import yaml
from typing import Dict, Any

def load_config(config_file: str = 'configs/algorithms/plot_config.yaml'):
    if not os.path.exists(config_file):
        print(f"Config file {config_file} not found. Using default configuration.")
        return {
            'input_file': 'results/discrete/convergence/tsp_aco_convergence.csv',
            'output_dir': 'visualizations/discrete/convergence',
            'plot_dpi': 300,
            'figure_size': [10, 6]
        }

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        print("Invalid config file format. Using default configuration.")
        config = {
            'input_file': 'results/discrete/convergence/tsp_aco_convergence.csv',
            'output_dir': 'visualizations/discrete/convergence',
            'plot_dpi': 300,
            'figure_size': [10, 6]
        }

    print(f"Loaded configuration from {config_file}")
    return config

def load_convergence_data(csv_file: str):
    if not os.path.exists(csv_file):
        print(f"CSV file {csv_file} not found.")
        return None, [], []

    with open(csv_file, 'r') as f:
        reader = list(csv.reader(f))

    if len(reader) < 5:
        print("CSV file does not contain enough data.")
        return None, [], []

    problem_info = {
        'name': reader[0][1] if len(reader[0]) > 1 else 'Unknown',
        'cities': int(reader[1][1]) if len(reader[1]) > 1 else 0,
        'best_length': float(reader[2][1]) if len(reader[2]) > 1 else 0.0
    }

    iterations, best_lengths = [], []
    for row in reader[5:]:
        if len(row) >= 2 and row[0].isdigit():
            iterations.append(int(row[0]))
            try:
                best_lengths.append(float(row[1]))
            except ValueError:
                best_lengths.append(0.0)

    return problem_info, iterations, best_lengths

def plot_convergence(problem_info, iterations, best_lengths, save_path: str, config: Dict[str, Any]):
    if not problem_info or not iterations or not best_lengths:
        print("No data to plot.")
        return

    plt.figure(figsize=tuple(config.get('figure_size', [10, 6])))
    plt.plot(iterations, best_lengths, 'b-', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Best Tour Length')
    plt.title(f"ACO Convergence - {problem_info['name']}\n"
              f"Best: {problem_info['best_length']:.2f}, Cities: {problem_info['cities']}")
    plt.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=config.get('plot_dpi', 300), bbox_inches='tight')
    print(f"Convergence plot saved to {save_path}")
    plt.close()

def plot_convergence_from_file(
    input_file: str = 'results/discrete/convergence/tsp_aco_convergence.csv',
    output_dir: str = 'visualizations/discrete/convergence',
    config_file: str = 'configs/algorithms/plot_config.yaml'
):
    """Plot convergence curve from CSV file."""
    config = load_config(config_file)
    
    # Override with provided paths
    if input_file:
        config['input_file'] = input_file
    if output_dir:
        config['output_dir'] = output_dir
    
    input_path = config.get('input_file', 'results/discrete/convergence/tsp_aco_convergence.csv')

    if not os.path.exists(input_path):
        print(f"Input file {input_path} not found.")
        print("Please run ACO optimization first (main.py > Discrete Problems > Run ACO)")
        return

    problem_info, iterations, best_lengths = load_convergence_data(input_path)

    if not problem_info:
        print("Failed to load convergence data.")
        return

    output_path = config.get('output_dir', 'visualizations/discrete/convergence')
    os.makedirs(output_path, exist_ok=True)

    plot_file = os.path.join(output_path, f"{problem_info['name']}_convergence.png")
    plot_convergence(problem_info, iterations, best_lengths, plot_file, config)
    
    print(f"\nConvergence plot created successfully!")


def main():
    plot_convergence_from_file()

if __name__ == "__main__":
    main()
