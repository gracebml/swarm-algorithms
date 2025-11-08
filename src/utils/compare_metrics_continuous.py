import numpy as np
import pandas as pd
import time
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import optimizers
from optimizers.continuous.ga_optimizer import GeneticAlgorithm
from optimizers.continuous.fa_optimizer import FireflyAlgorithm
from optimizers.continuous.pso_optimizer import pso
from optimizers.continuous.sa_optimizer import SimulatedAnnealing
from optimizers.continuous.abc_optimizter import ArtificialBeeColony
from optimizers.continuous.cs_optimizer import CuckooSearch

# Import problems
from problems.continuous.sphere_function import SphereFunction
from problems.continuous.ackley_function import AckleyFunction
from problems.continuous.rastrigin_function import RastriginFunction

# Import config loader
try:
    from utils.config_loader import load_algorithm_params
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False


def run_algorithm_multiple_times(algorithm_name, problem, n_runs=30, **alg_params):

    results = {
        'best_fitness': [],
        'convergence_curves': [],
        'execution_times': [],
        'memory_usage': []
    }
    
    bounds = problem.get_bounds()
    
    for run in range(n_runs):
        start_time = time.time()
        
        if algorithm_name.upper() == 'GA':
            ga = GeneticAlgorithm(
                objective_function=problem.evaluate,
                bounds=bounds,
                random_seed=42 + run,
                **alg_params
            )
            best_solution, best_fitness = ga.optimize(verbose=False)
            history = ga.get_history()
            convergence_curve = history['best_fitness']
            
        elif algorithm_name.upper() == 'FA':
            fa = FireflyAlgorithm(
                objective_function=problem.evaluate,
                bounds=bounds,
                random_seed=42 + run,
                **alg_params
            )
            best_solution, best_fitness = fa.optimize(verbose=False)
            history = fa.get_history()
            convergence_curve = history['best_fitness']
            
        elif algorithm_name.upper() == 'PSO':
            min_x = bounds[0][0]
            max_x = bounds[0][1]
            dim = len(bounds)
            result = pso(
                fitness=problem.evaluate,
                dim=dim,
                min_x=min_x,
                max_x=max_x,
                minimize=True,
                seed=42 + run,
                **alg_params
            )
            best_fitness = result['best_fitness']
            convergence_curve = result['fit_res']
            
        elif algorithm_name.upper() == 'SA':
            sa = SimulatedAnnealing(
                objective_function=problem.evaluate,
                bounds=bounds,
                random_seed=42 + run,
                **alg_params
            )
            best_solution, best_fitness = sa.optimize(verbose=False)
            history = sa.get_history()
            convergence_curve = history['best_fitness']
            
        elif algorithm_name.upper() == 'CS':
            min_x = bounds[0][0]
            max_x = bounds[0][1]
            dim = len(bounds)
            
            cs = CuckooSearch(seed=42 + run, **alg_params)
            result = cs.optimize(
                objective_func=problem.evaluate,
                dim=dim,
                bounds=(min_x, max_x),
                minimize=True,
                verbose=False
            )
            best_fitness = result['best_fitness']
            convergence_curve = result['history']
            
        elif algorithm_name.upper() == 'ABC':
            abc = ArtificialBeeColony(
                objective_function=problem,
                verbose=False,
                **alg_params
            )
            result = abc.optimize()
            best_fitness = result['best_value']
            convergence_curve = result['history']['best_value']
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        
        end_time = time.time()
        
        results['best_fitness'].append(best_fitness)
        results['convergence_curves'].append(convergence_curve)
        results['execution_times'].append(end_time - start_time)
    
    return results


def compute_convergence_speed(convergence_curves, threshold=0.99):
    convergence_iterations = []
    
    for curve in convergence_curves:
        curve = np.array(curve)
        if len(curve) == 0:
            continue
            
        start_val = curve[0]
        end_val = curve[-1]
        target_val = start_val - threshold * (start_val - end_val)
        
        # Find when target is reached
        converged_at = len(curve)
        for i, val in enumerate(curve):
            if val <= target_val:
                converged_at = i
                break
        
        convergence_iterations.append(converged_at)
    
    return np.mean(convergence_iterations) if convergence_iterations else None


def compute_robustness(fitness_values):
    std = np.std(fitness_values)
    return 1.0 / (1.0 + std)


def compute_scalability(algorithm_name, problem_class, dimensions_list, n_runs=5, **alg_params):
    times = []
    
    for dim in dimensions_list:
        problem = problem_class(dimensions=dim)
        result = run_algorithm_multiple_times(
            algorithm_name,
            problem,
            n_runs=n_runs,
            **alg_params
        )
        avg_time = np.mean(result['execution_times'])
        times.append(avg_time)
    
    # Fit linear model: time = a * dim + b
    # Return slope (a)
    coeffs = np.polyfit(dimensions_list, times, 1)
    return coeffs[0]  # slope


def benchmark_algorithms(algorithms_config, problem, n_runs=30, output_dir='results/performance'):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    problem_name = problem.__class__.__name__.replace('Function', '')
    
    print(f"\n{'='*70}")
    print(f"BENCHMARKING ALGORITHMS ON {problem_name.upper()} FUNCTION")
    print(f"{'='*70}")
    print(f"Problem dimensions: {problem.dimensions}")
    print(f"Number of runs: {n_runs}\n")
    
    results_rows = []
    detailed_results = {}
    
    for alg_name, alg_params in algorithms_config.items():
        print(f"Running {alg_name}...")
        
        # Run algorithm multiple times
        results = run_algorithm_multiple_times(
            alg_name,
            problem,
            n_runs=n_runs,
            **alg_params
        )
        
        # Compute metrics
        best_fitness = np.min(results['best_fitness'])
        mean_fitness = np.mean(results['best_fitness'])
        std_fitness = np.std(results['best_fitness'])
        mean_time = np.mean(results['execution_times'])
        std_time = np.std(results['execution_times'])
        
        conv_speed = compute_convergence_speed(results['convergence_curves'])
        robustness = compute_robustness(results['best_fitness'])
        
        # Store detailed results
        detailed_results[alg_name] = {
            'best_fitness_values': results['best_fitness'],
            'convergence_curves': results['convergence_curves'],
            'execution_times': results['execution_times']
        }
        
        results_rows.append({
            'Algorithm': alg_name,
            'Best Fitness': best_fitness,
            'Mean Fitness': mean_fitness,
            'Std Fitness': std_fitness,
            'Convergence Speed (iter)': conv_speed,
            'Mean Time (s)': mean_time,
            'Std Time (s)': std_time,
            'Robustness': robustness
        })
        
        print(f"  Best: {best_fitness:.10f}, Mean: {mean_fitness:.10f}, Time: {mean_time:.3f}s")
    
    # Create DataFrame
    df = pd.DataFrame(results_rows)
    
    # Save summary to CSV
    csv_file = output_path / f'{problem_name.lower()}_metrics.csv'
    df.to_csv(csv_file, index=False)
    print(f"\n Saved metrics to {csv_file}")
    
    # Save detailed results to JSON
    json_file = output_path / f'{problem_name.lower()}_detailed.json'
    with open(json_file, 'w') as f:
        json.dump(detailed_results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    print(f" Saved detailed results to {json_file}")
    
    return df, detailed_results


def benchmark_scalability(algorithms_config, problem_class, dimensions_list=[5, 10, 20, 30], 
                          n_runs=5, output_dir='results/performance'):  
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    problem_name = problem_class.__name__.replace('Function', '')
    
    print(f"\n{'='*70}")
    print(f"SCALABILITY ANALYSIS ON {problem_name.upper()} FUNCTION")
    print(f"{'='*70}")
    print(f"Testing dimensions: {dimensions_list}")
    print(f"Runs per dimension: {n_runs}\n")
    
    scalability_data = {alg_name: {'dimensions': [], 'times': []} 
                        for alg_name in algorithms_config.keys()}
    
    for dim in dimensions_list:
        print(f"\nDimension: {dim}")
        problem = problem_class(dimensions=dim)
        
        for alg_name, alg_params in algorithms_config.items():
            print(f"  Testing {alg_name}...", end=' ')
            
            results = run_algorithm_multiple_times(
                alg_name,
                problem,
                n_runs=n_runs,
                **alg_params
            )
            
            avg_time = np.mean(results['execution_times'])
            scalability_data[alg_name]['dimensions'].append(dim)
            scalability_data[alg_name]['times'].append(avg_time)
            
            print(f"Time: {avg_time:.3f}s")
    
    # Compute scalability coefficients
    results_rows = []
    for alg_name, data in scalability_data.items():
        coeffs = np.polyfit(data['dimensions'], data['times'], 1)
        slope = coeffs[0]
        
        results_rows.append({
            'Algorithm': alg_name,
            'Scalability Coefficient': slope,
            'Time Complexity': 'O(n)' if slope < 0.1 else 'O(n^2)' if slope < 1 else 'O(n^3)'
        })
    
    df = pd.DataFrame(results_rows)
    
    # Save results
    csv_file = output_path / f'{problem_name.lower()}_scalability.csv'
    df.to_csv(csv_file, index=False)
    print(f"\n Saved scalability metrics to {csv_file}")
    
    # Save detailed data
    json_file = output_path / f'{problem_name.lower()}_scalability_data.json'
    with open(json_file, 'w') as f:
        json.dump(scalability_data, f, indent=2)
    print(f" Saved scalability data to {json_file}")
    
    return df, scalability_data
