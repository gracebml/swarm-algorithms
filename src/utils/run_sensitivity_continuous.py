import numpy as np
import json
import yaml
import pandas as pd
from pathlib import Path
import sys

# Import all optimizers
sys.path.insert(0, str(Path(__file__).parent.parent))
from optimizers.continuous.ga_optimizer import GeneticAlgorithm
from optimizers.continuous.fa_optimizer import FireflyAlgorithm
from optimizers.continuous.pso_optimizer import pso
from optimizers.continuous.sa_optimizer import SimulatedAnnealing
from optimizers.continuous.cs_optimizer import CuckooSearch
from optimizers.continuous.abc_optimizter import ArtificialBeeColony

# Import problem functions
from problems.continuous.sphere_function import SphereFunction
from problems.continuous.ackley_function import AckleyFunction
from problems.continuous.rastrigin_function import RastriginFunction


def get_problem_instance(problem_name, dimensions):
    """Get problem instance by name."""
    problems = {
        'sphere': SphereFunction,
        'ackley': AckleyFunction,
        'rastrigin': RastriginFunction
    }
    
    if problem_name.lower() not in problems:
        raise ValueError(f"Unknown problem: {problem_name}")
    
    return problems[problem_name.lower()](dimensions=dimensions)


def run_single_experiment_ga(problem, params, n_runs=30, generations=100, seed_base=42):
    """Run GA sensitivity experiment."""
    results = {
        'best_fitness': [],
        'mean_fitness': [],
        'std_fitness': [],
        'convergence_curves': []
    }
    
    bounds = problem.get_bounds()
    
    for run in range(n_runs):
        ga = GeneticAlgorithm(
            objective_function=problem.evaluate,
            bounds=bounds,
            population_size=params.get('population_size', 50),
            generations=generations,
            crossover_rate=params.get('crossover_rate', 0.8),
            mutation_rate=params.get('mutation_rate', 0.1),
            elite_size=params.get('elite_size', 2),
            tournament_size=params.get('tournament_size', 3),
            random_seed=seed_base + run
        )
        
        best_solution, best_fitness = ga.optimize(verbose=False)
        history = ga.get_history()
        
        results['best_fitness'].append(best_fitness)
        results['mean_fitness'].append(history['mean_fitness'][-1])
        results['std_fitness'].append(np.std(history['mean_fitness']))
        results['convergence_curves'].append(history['best_fitness'])
    
    return results


def run_single_experiment_fa(problem, params, n_runs=30, generations=100, seed_base=42):
    """Run FA sensitivity experiment."""
    results = {
        'best_fitness': [],
        'mean_fitness': [],
        'std_fitness': [],
        'convergence_curves': []
    }
    
    bounds = problem.get_bounds()
    
    for run in range(n_runs):
        fa = FireflyAlgorithm(
            objective_function=problem.evaluate,
            bounds=bounds,
            population_size=params.get('population_size', 50),
            generations=generations,
            alpha=params.get('alpha', 0.5),
            beta0=params.get('beta0', 1.0),
            gamma=params.get('gamma', 1.0),
            random_seed=seed_base + run
        )
        
        best_solution, best_fitness = fa.optimize(verbose=False)
        history = fa.get_history()
        
        results['best_fitness'].append(best_fitness)
        results['mean_fitness'].append(history['mean_fitness'][-1])
        results['std_fitness'].append(np.std(history['mean_fitness']))
        results['convergence_curves'].append(history['best_fitness'])
    
    return results


def run_single_experiment_pso(problem, params, n_runs=30, max_iter=100, seed_base=42):
    """Run PSO sensitivity experiment."""
    results = {
        'best_fitness': [],
        'mean_fitness': [],
        'std_fitness': [],
        'convergence_curves': []
    }
    
    bounds = problem.get_bounds()
    min_x = bounds[0][0]
    max_x = bounds[0][1]
    dim = len(bounds)
    
    for run in range(n_runs):
        result = pso(
            fitness=problem.evaluate,
            max_iter=max_iter,
            n=params.get('N', 50),
            dim=dim,
            min_x=min_x,
            max_x=max_x,
            w=params.get('w', 0.95),
            c1=params.get('c1', 1.49445),
            c2=params.get('c2', 1.49445),
            minimize=True,
            seed=seed_base + run
        )
        
        results['best_fitness'].append(result['best_fitness'])
        results['mean_fitness'].append(result['best_fitness'])  # PSO doesn't track mean
        results['std_fitness'].append(0)
        results['convergence_curves'].append(result['fit_res'])
    
    return results


def run_single_experiment_sa(problem, params, n_runs=30, max_iterations=100, seed_base=42):
    """Run SA sensitivity experiment."""
    results = {
        'best_fitness': [],
        'mean_fitness': [],
        'std_fitness': [],
        'convergence_curves': []
    }
    
    bounds = problem.get_bounds()
    
    for run in range(n_runs):
        sa = SimulatedAnnealing(
            objective_function=problem.evaluate,
            bounds=bounds,
            max_iterations=max_iterations,
            initial_temp=params.get('initial_temp', 100),
            cooling_rate=params.get('cooling_rate', 0.95),
            neighbor_scale=params.get('neighbor_scale', 0.1),
            random_seed=seed_base + run
        )
        
        best_solution, best_fitness = sa.optimize(verbose=False)
        history = sa.get_history()
        
        results['best_fitness'].append(best_fitness)
        results['mean_fitness'].append(best_fitness)  # SA doesn't have population
        results['std_fitness'].append(0)
        results['convergence_curves'].append(history['best_fitness'])
    
    return results


def run_single_experiment_cs(problem, params, n_runs=30, max_iter=100, seed_base=42):
    """Run CS sensitivity experiment."""
    results = {
        'best_fitness': [],
        'mean_fitness': [],
        'std_fitness': [],
        'convergence_curves': []
    }
    
    bounds = problem.get_bounds()
    min_x = bounds[0][0]
    max_x = bounds[0][1]
    dim = len(bounds)
    
    for run in range(n_runs):
        cs = CuckooSearch(
            n_nests=params.get('n_nests', 25),
            pa=params.get('pa', 0.25),
            beta=params.get('beta', 1.5),
            alpha=params.get('alpha', 0.01),
            seed=seed_base + run
        )
        
        result = cs.optimize(
            objective_func=problem.evaluate,
            dim=dim,
            bounds=(min_x, max_x),
            max_iter=max_iter,
            minimize=True,
            verbose=False
        )
        
        results['best_fitness'].append(result['best_fitness'])
        results['mean_fitness'].append(result['mean_history'][-1])
        results['std_fitness'].append(np.std(result['fitness']))
        results['convergence_curves'].append(result['history'])
    
    return results


def run_single_experiment_abc(problem, params, n_runs=30, max_iterations=100, seed_base=42):
    """Run ABC sensitivity experiment."""
    results = {
        'best_fitness': [],
        'mean_fitness': [],
        'std_fitness': [],
        'convergence_curves': []
    }
    
    # Calculate limit from limit_multiplier if provided
    colony_size = params.get('colony_size', 50)
    dimensions = problem.dimensions
    
    if 'limit_multiplier' in params:
        limit = int(colony_size * dimensions * params['limit_multiplier'])
    else:
        limit = params.get('limit', None)  # Will use default if None
    
    for run in range(n_runs):
        # Set random seed for reproducibility
        np.random.seed(seed_base + run)
        
        abc = ArtificialBeeColony(
            objective_function=problem,
            colony_size=colony_size,
            max_iterations=max_iterations,
            limit=limit,
            verbose=False
        )
        
        abc.initialize_population()
        result = abc.optimize()
        
        results['best_fitness'].append(result['best_value'])
        results['mean_fitness'].append(result['history']['mean_value'][-1])
        results['std_fitness'].append(np.std(result['history']['mean_value']))
        results['convergence_curves'].append(result['history']['best_value'])
    
    return results


def sensitivity_analysis_single_param(algorithm_name, problem_name, param_name, param_values, 
                                     base_params, config, output_dir='results/continuous/parameter_sensitivity'):
    """
    Run sensitivity analysis for a single parameter.
    
    Args:
        algorithm_name: Name of algorithm (GA, FA, PSO, SA, CS)
        problem_name: Name of problem function
        param_name: Parameter to vary
        param_values: List of values to test
        base_params: Base parameter configuration
        config: Run configuration (n_runs, dimensions, etc.)
        output_dir: Output directory
    """
    problem = get_problem_instance(problem_name, config['dimensions'])
    
    # Create output directory
    output_path = Path(output_dir) / algorithm_name.lower()
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_summary = []
    
    print(f"\n{'='*70}")
    print(f"Algorithm: {algorithm_name} | Problem: {problem_name} | Parameter: {param_name}")
    print(f"{'='*70}")
    
    for value in param_values:
        params = base_params.copy()
        params[param_name] = value
        
        print(f"\nTesting {param_name} = {value}...")
        
        # Select appropriate runner
        if algorithm_name.upper() == 'GA':
            results = run_single_experiment_ga(
                problem, params, 
                n_runs=config['n_runs'],
                generations=config.get('generations', 100),
                seed_base=config.get('seed', 42)
            )
        elif algorithm_name.upper() == 'FA':
            results = run_single_experiment_fa(
                problem, params,
                n_runs=config['n_runs'],
                generations=config.get('generations', 100),
                seed_base=config.get('seed', 42)
            )
        elif algorithm_name.upper() == 'PSO':
            results = run_single_experiment_pso(
                problem, params,
                n_runs=config['n_runs'],
                max_iter=config.get('max_iter', 100),
                seed_base=config.get('seed', 42)
            )
        elif algorithm_name.upper() == 'SA':
            results = run_single_experiment_sa(
                problem, params,
                n_runs=config['n_runs'],
                max_iterations=config.get('max_iterations', 100),
                seed_base=config.get('seed', 42)
            )
        elif algorithm_name.upper() == 'CS':
            results = run_single_experiment_cs(
                problem, params,
                n_runs=config['n_runs'],
                max_iter=config.get('max_iter', 100),
                seed_base=config.get('seed', 42)
            )
        elif algorithm_name.upper() == 'ABC':
            results = run_single_experiment_abc(
                problem, params,
                n_runs=config['n_runs'],
                max_iterations=config.get('max_iterations', 100),
                seed_base=config.get('seed', 42)
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")
        
        # Calculate statistics
        best_mean = np.mean(results['best_fitness'])
        best_std = np.std(results['best_fitness'])
        best_min = np.min(results['best_fitness'])
        best_max = np.max(results['best_fitness'])
        
        # Average convergence curve across runs
        avg_convergence = np.mean(results['convergence_curves'], axis=0)
        
        results_summary.append({
            'algorithm': algorithm_name,
            'function': problem_name,
            'parameter': param_name,
            'value': value,
            'best_mean': best_mean,
            'best_std': best_std,
            'best_min': best_min,
            'best_max': best_max,
            'convergence': avg_convergence.tolist()
        })
        
        print(f"  Best Mean: {best_mean:.10f} ± {best_std:.10f}")
        print(f"  Best Min: {best_min:.10f}")
        print(f"  Best Max: {best_max:.10f}")
    
    # Save results to CSV
    df = pd.DataFrame([{
        'algorithm': r['algorithm'],
        'function': r['function'],
        'parameter': r['parameter'],
        'value': r['value'],
        'best_mean': r['best_mean'],
        'best_std': r['best_std'],
        'best_min': r['best_min'],
        'best_max': r['best_max']
    } for r in results_summary])
    
    csv_file = output_path / f'{algorithm_name.lower()}_{problem_name}_{param_name}_sensitivity.csv'
    df.to_csv(csv_file, index=False)
    print(f"\nSaved results to {csv_file}")
    
    # Save convergence curves separately
    convergence_data = {
        'parameter_values': param_values,
        'convergence_curves': [r['convergence'] for r in results_summary]
    }
    
    json_file = output_path / f'{algorithm_name.lower()}_{problem_name}_{param_name}_convergence.json'
    with open(json_file, 'w') as f:
        json.dump(convergence_data, f, indent=2)
    print(f"Saved convergence data to {json_file}")
    
    return results_summary


def run_sensitivity_from_config(config_file):
    """
    Run sensitivity analysis from a YAML configuration file.
    
    Args:
        config_file: Path to YAML config file
    """
    print(f"\nLoading configuration from: {config_file}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    algorithm_name = config['algorithm']['name']
    problems = config['problems']
    sensitivity_params = config['sensitivity']
    n_runs = config['n_runs']
    output_dir = config.get('output_directory', 'results/continuous/parameter_sensitivity')
    
    # Extract base parameters
    base_params = {k: v for k, v in config['algorithm'].items() if k != 'name'}
    
    print(f"\n{'='*70}")
    print(f"SENSITIVITY ANALYSIS: {algorithm_name}")
    print(f"{'='*70}")
    print(f"\nBase Parameters:")
    for key, val in base_params.items():
        print(f"  {key}: {val}")
    
    print(f"\nProblems: {list(problems.keys())}")
    print(f"Number of runs per configuration: {n_runs}")
    print(f"Parameters to test: {list(sensitivity_params.keys())}")
    
    # Run sensitivity analysis for each problem and parameter
    for problem_name, problem_config in problems.items():
        for param_name, param_values in sensitivity_params.items():
            # Create run config
            run_config = {
                'n_runs': n_runs,
                'dimensions': problem_config['dimensions'],
                'seed': base_params.get('random_seed', 42)
            }
            
            # Add algorithm-specific config
            if 'generations' in base_params:
                run_config['generations'] = base_params['generations']
            if 'max_iter' in base_params:
                run_config['max_iter'] = base_params['max_iter']
            if 'max_iterations' in base_params:
                run_config['max_iterations'] = base_params['max_iterations']
            
            sensitivity_analysis_single_param(
                algorithm_name=algorithm_name,
                problem_name=problem_name,
                param_name=param_name,
                param_values=param_values,
                base_params=base_params,
                config=run_config,
                output_dir=output_dir
            )
    
    print(f"\n{'='*70}")
    print(f"SENSITIVITY ANALYSIS COMPLETE!")
    print(f"{'='*70}")
    print(f"Results saved to: {output_dir}")


def run_all_sensitivity_analyses():
    """Run sensitivity analysis for all algorithms."""
    config_dir = Path('configs/sensitivity')
    
    config_files = [
        config_dir / 'ga_sensitivity.yaml',
        config_dir / 'fa_sensitivity.yaml',
        config_dir / 'pso_sensitivity.yaml',
        config_dir / 'sa_sensitivity.yaml',
        config_dir / 'cs_sensitivity.yaml',
        config_dir / 'abc_sensitivity.yaml',
    ]
    
    for config_file in config_files:
        if config_file.exists():
            print(f"\n{'#'*70}")
            print(f"# Processing: {config_file.name}")
            print(f"{'#'*70}")
            run_sensitivity_from_config(config_file)
        else:
            print(f"\nWarning: Config file not found: {config_file}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run parameter sensitivity analysis')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file (if not provided, runs all)')
    parser.add_argument('--algorithm', type=str, default=None,
                       help='Algorithm name (ga, fa, pso, sa, cs) to run specific algorithm')
    
    args = parser.parse_args()
    
    if args.config:
        run_sensitivity_from_config(args.config)
    elif args.algorithm:
        config_file = f'configs/sensitivity/{args.algorithm.lower()}_sensitivity.yaml'
        run_sensitivity_from_config(config_file)
    else:
        run_all_sensitivity_analyses()
