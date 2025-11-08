import numpy as np
import copy
import os
import csv
import yaml
import itertools
import sys
from pathlib import Path

# Ensure src is in path for imports
current_file = Path(__file__).resolve()
src_dir = current_file.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from problems.discrete.tsp import TSPProblem
from optimizers.discrete.aco import MMAS, load_config


def run_sensitivity_analysis(config_file):
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    problem_path = config.get('problem_file')
    
    # Resolve path relative to project root
    if problem_path and not os.path.isabs(problem_path):
        # Get project root (3 levels up from src/utils/)
        project_root = Path(__file__).resolve().parent.parent.parent
        problem_path = str(project_root / problem_path)
    
    if not problem_path or not os.path.exists(problem_path):
        raise FileNotFoundError(f"Problem file not found: {problem_path}")

    problem = TSPProblem(problem_file=problem_path)
    base_config = load_config()

    base_n_ants = int(base_config.get('n_ants', 20))
    if base_n_ants == -1:
        base_n_ants = problem.dim

    n_ants_values = [int(base_n_ants * mult) for mult in config['sensitivity']['n_ants']]
    param_grid = {
        'n_ants': n_ants_values,
        'max_iter': config['sensitivity']['max_iter'],
        'alpha': config['sensitivity']['alpha'],
        'beta': config['sensitivity']['beta'],
        'rho': config['sensitivity']['rho']
    }

    param_keys = list(param_grid.keys())
    all_combinations = list(itertools.product(*param_grid.values()))
    total_combos = len(all_combinations)

    analysis_results = []

    for combo_values in all_combinations:
        current_config = copy.deepcopy(base_config)
        combo_dict = dict(zip(param_keys, combo_values))
        current_config.update(combo_dict)
        run_fitnesses = []

        for _ in range(config['num_runs']):
            mmas = MMAS(problem, current_config)
            result = mmas.optimize()
            run_fitnesses.append(result['best_fitness'])

        avg_fitness = np.mean(run_fitnesses)
        std_fitness = np.std(run_fitnesses)
        analysis_results.append({
            'params': combo_dict,
            'avg_fitness': avg_fitness,
            'std_fitness': std_fitness,
            'all_fitnesses': run_fitnesses
        })

    output_dir = config.get('output_directory', 'results/discrete/parameter_sensitivity/aco')
    
    # Resolve output path relative to project root
    if not os.path.isabs(output_dir):
        project_root = Path(__file__).resolve().parent.parent.parent
        output_dir = str(project_root / output_dir)
    
    os.makedirs(output_dir, exist_ok=True)

    results_file = os.path.join(output_dir, 'sensitivity_results.csv')
    with open(results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(param_keys + ['avg_fitness', 'std_fitness', 'all_fitnesses'])
        for result in analysis_results:
            row = [result['params'][key] for key in param_keys]
            row.extend([result['avg_fitness'], result['std_fitness'], str(result['all_fitnesses'])])
            writer.writerow(row)

    detailed_file = os.path.join(output_dir, 'sensitivity_detailed.yaml')
    with open(detailed_file, 'w') as f:
        yaml.dump(analysis_results, f, default_flow_style=False)

    grid_file = os.path.join(output_dir, 'parameter_grid.yaml')
    with open(grid_file, 'w') as f:
        yaml.dump({
            'param_grid': param_grid,
            'total_combinations': total_combos,
            'num_runs_per_combo': config['num_runs'],
            'base_n_ants': base_n_ants
        }, f, default_flow_style=False)

    # Create individual parameter sensitivity files for visualization
    for param_name in param_keys:
        # Get unique values for this parameter
        param_specific_results = {}
        for result in analysis_results:
            param_val = result['params'][param_name]
            if param_val not in param_specific_results:
                param_specific_results[param_val] = []
            param_specific_results[param_val].append(result['avg_fitness'])
        
        # Calculate mean and std for each parameter value
        param_file = os.path.join(output_dir, f'aco_{param_name}_sensitivity.csv')
        with open(param_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([param_name, 'mean_fitness', 'std_fitness'])
            for param_val in sorted(param_specific_results.keys()):
                fitnesses = param_specific_results[param_val]
                mean_fit = np.mean(fitnesses)
                std_fit = np.std(fitnesses)
                writer.writerow([param_val, mean_fit, std_fit])
        
        print(f"  Created: {param_file}")

    return analysis_results


# if __name__ == "__main__":
#     run_sensitivity_analysis("../../configs/aco_sensitivity_config.yaml")