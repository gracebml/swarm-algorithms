import csv
import os
import yaml
import time
import tracemalloc
import numpy as np
import sys
from pathlib import Path

# Ensure src is in path for imports
current_file = Path(__file__).resolve()
src_dir = current_file.parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from problems.discrete.tsp import TSPProblem
from optimizers.discrete.aco import MMAS, load_config as load_aco_config
from optimizers.discrete.astar import TSPAStarSolver

def load_compare_config(config_file):
    if not os.path.exists(config_file):
        return None
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def load_astar_config(config_file):
    if not os.path.exists(config_file):
        return None
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def calculate_scalability(dimensions, times):
    if len(dimensions) < 2:
        return 0.0
    return np.polyfit(dimensions, times, 1)[0]

def run_scalability_analysis(problem_files, num_runs, aco_config, astar_config):
    mmas_dimensions, mmas_times, astar_dimensions, astar_times = [], [], [], []
    for problem_file in problem_files:
        if not os.path.exists(problem_file):
            continue
        problem = TSPProblem(problem_file=problem_file)
        mmas_run_times, astar_run_times = [], []
        for _ in range(num_runs):
            mmas = MMAS(problem, aco_config)
            t0 = time.time()
            mmas.optimize()
            mmas_run_times.append(time.time() - t0)
        for _ in range(num_runs):
            solver = TSPAStarSolver(problem, astar_config.get('time_limit', 3600.0))
            t0 = time.time()
            solver.solve(astar_config.get('start_city', 0))
            astar_run_times.append(time.time() - t0)
        mmas_dimensions.append(problem.dim)
        mmas_times.append(np.mean(mmas_run_times))
        astar_dimensions.append(problem.dim)
        astar_times.append(np.mean(astar_run_times))
    mmas_scalability = calculate_scalability(mmas_dimensions, mmas_times)
    astar_scalability = calculate_scalability(astar_dimensions, astar_times)
    return mmas_scalability, astar_scalability

def run_tsp_comparison(config_file='configs/compare_config.yaml'):
    """Run TSP comparison between ACO and A* algorithms."""
    compare_config = load_compare_config(config_file)
    if not compare_config:
        return
    problem_file = compare_config.get('problem_file')
    aco_config_file = compare_config.get('aco_config_file')
    astar_config_file = compare_config.get('astar_config_file')
    comparison_output_file = compare_config.get('comparison_output_file')
    num_runs = int(compare_config.get('num_runs', 10))
    scalability_problems = compare_config.get('scalability_problems', [])
    if not problem_file or not os.path.exists(problem_file):
        return
    problem = TSPProblem(problem_file=problem_file)
    aco_config = load_aco_config(aco_config_file)
    astar_config = load_astar_config(astar_config_file)
    if not aco_config or not astar_config:
        return
    mmas_fitness_results, mmas_time_results, astar_fitness_results, astar_time_results = [], [], [], []
    mmas_peak_mem, astar_peak_mem = 0, 0
    if scalability_problems:
        mmas_scalability_coef, astar_scalability_coef = run_scalability_analysis(
            scalability_problems, min(num_runs, 3), aco_config, astar_config
        )
    else:
        mmas_scalability_coef = astar_scalability_coef = 0.0
    for i in range(num_runs):
        if i == 0:
            tracemalloc.start()
        mmas = MMAS(problem, aco_config)
        t0 = time.time()
        mmas_result = mmas.optimize()
        t_mmas = time.time() - t0
        if i == 0:
            mmas_peak_mem = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
        mmas_fitness_results.append(mmas_result['best_fitness'])
        mmas_time_results.append(t_mmas)
        if i == 0:
            tracemalloc.start()
        solver = TSPAStarSolver(problem, astar_config.get('time_limit', 3600.0))
        t0 = time.time()
        astar_result = solver.solve(astar_config.get('start_city', 0))
        t_astar = time.time() - t0
        if i == 0:
            astar_peak_mem = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
        astar_fitness_results.append(astar_result['cost'])
        astar_time_results.append(t_astar)
    avg_time_mmas = np.mean(mmas_time_results)
    avg_fitness_mmas = np.mean(mmas_fitness_results)
    var_fitness_mmas = np.var(mmas_fitness_results)
    if avg_fitness_mmas == 0:
        robustness_str_mmas = "N/A"
    elif var_fitness_mmas == 0:
        robustness_str_mmas = "1.0"
    else:
        robustness_str_mmas = f"{1 - (var_fitness_mmas / avg_fitness_mmas):.4f}"
    avg_time_astar = np.mean(astar_time_results)
    avg_fitness_astar = np.mean(astar_fitness_results)
    var_fitness_astar = np.var(astar_fitness_results)
    if avg_fitness_astar == 0:
        robustness_str_astar = "N/A"
    elif var_fitness_astar == 0:
        robustness_str_astar = "1.0"
    else:
        robustness_str_astar = f"{1 - (var_fitness_astar / avg_fitness_astar):.4f}"
    header = [
        'Algorithm',
        'Avg Convergence Speed',
        'Memory Usage',
        'Avg Tour Length',
        'Robustness',
        'Scalability Coefficient'
    ]
    data = [
        [
            'MMAS (ACO)',
            f"{avg_time_mmas:.6f}",
            f"{mmas_peak_mem}",
            f"{avg_fitness_mmas:.6f}",
            robustness_str_mmas,
            f"{mmas_scalability_coef:.6f}"
        ],
        [
            'A* (MST)',
            f"{avg_time_astar:.6f}",
            f"{astar_peak_mem}",
            f"{avg_fitness_astar:.6f}",
            robustness_str_astar,
            f"{astar_scalability_coef:.6f}"
        ]
    ]
    output_dir = os.path.dirname(comparison_output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(comparison_output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)

# if __name__ == "__main__":
#     main()