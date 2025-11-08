"""
Discrete optimization problems menu and functions (TSP with ACO, A*)
"""

import sys
import os
from pathlib import Path

# Note: Path setup is done by main.py, so imports work when running via main.py
# If running this file directly for testing, uncomment the path setup below:
# src_dir = Path(__file__).parent.parent / 'src'
# if str(src_dir) not in sys.path:
#     sys.path.insert(0, str(src_dir))

# Discrete visualizations
from src.visualize.discrete.visualize_sensitivity import visualize_algorithm_results as visualize_discrete_sensitivity
from src.visualize.discrete.visualize_performance_comparison import visualize_performance_comparison
from src.visualize.discrete.run_plot_convergence import plot_convergence_from_file

# Utils
from src.utils.compare_metrics_discrete import run_tsp_comparison

# Problems and optimizers
from src.problems.discrete.tsp import TSPProblem
from src.optimizers.discrete.aco import MMAS, load_config as load_aco_config
from src.optimizers.discrete.astar import TSPAStarSolver

import csv


def discrete_menu():
    """Menu for discrete optimization problems (TSP)."""
    while True:
        print("\n" + "="*80)
        print(" DISCRETE PROBLEMS MENU - TSP")
        print("="*80)
        print("\n  1. Run ACO Optimization")
        print("  2. Run A* Search")
        print("  3. Performance Comparison (ACO vs A*)")
        print("  4. Convergence Visualization")
        print("  5. Sensitivity Analysis (ACO)")
        print("  0. Back to Main Menu")
        print("-" * 80)
        
        choice = input("\nChoice: ").strip()
        
        if choice == '0':
            break
        elif choice == '1':
            run_aco_optimization()
        elif choice == '2':
            run_astar_optimization()
        elif choice == '3':
            run_discrete_performance_comparison()
        elif choice == '4':
            run_discrete_convergence_visualization()
        elif choice == '5':
            run_discrete_sensitivity_analysis()
        else:
            print("\n Invalid choice! Please try again.")
        
        if choice != '0':
            input("\nPress Enter to continue...")


def run_aco_optimization():
    """Run ACO optimization for TSP."""
    print("\n" + "="*80)
    print(" ACO Optimization - TSP")
    print("="*80)
    
    # Load config
    config = load_aco_config('configs/algorithms/aco_config.yaml')
    problem_file = config.get('problem_file', 'data/tsp_10.csv')
    
    print(f"\n Problem file: {problem_file}")
    
    if not problem_file or not problem_file.endswith('.csv') or not os.path.exists(problem_file):
        print(f"\n Error: Problem file not found: {problem_file}")
        print(" Please check configs/algorithms/aco_config.yaml")
        return
    
    try:
        # Load problem
        problem = TSPProblem(problem_file=problem_file)
        print(f" Loaded: {problem.name} ({problem.dim} cities)")
        
        # Run ACO
        print(f"\n Running ACO optimization...")
        print(f"   n_ants: {config.get('n_ants', -1)} (-1 = auto)")
        print(f"   max_iter: {config.get('max_iter', 100)}")
        print(f"   alpha: {config.get('alpha', 1.0)}")
        print(f"   beta: {config.get('beta', 2.0)}")
        print(f"   rho: {config.get('rho', 0.1)}")
        
        mmas = MMAS(problem, config)
        result = mmas.optimize()
        
        print(f"\n" + "="*80)
        print(" ACO Optimization Results")
        print("="*80)
        print(f" Best tour length: {result['best_fitness']:.6f}")
        
        # Save results
        output_file = config.get('output_file', 'results/discrete/convergence/tsp_aco_convergence.csv')
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Problem', problem.name])
            writer.writerow(['Cities', problem.dim])
            writer.writerow(['Best Length', f"{result['best_fitness']:.6f}"])
            writer.writerow([])
            writer.writerow(['Iteration', 'Best_Length'])
            for i, length in enumerate(result['history']):
                writer.writerow([i, f"{length:.6f}"])
        
        print(f" Results saved to: {output_file}")
        print("="*80)
        
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()


def run_astar_optimization():
    """Run A* search for TSP."""
    print("\n" + "="*80)
    print(" A* Search - TSP")
    print("="*80)
    
    # Load config
    try:
        import yaml
        config_path = 'configs/algorithms/astar_config.yaml'
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        else:
            config = {
                'problem_file': 'data/tsp_10.csv',
                'time_limit': 3600.0,
                'start_city': 0
            }
    except Exception:
        config = {
            'problem_file': 'data/tsp_10.csv',
            'time_limit': 3600.0,
            'start_city': 0
        }
    
    problem_file = config.get('problem_file', 'data/tsp_10.csv')
    time_limit = config.get('time_limit', 3600.0)
    start_city = config.get('start_city', 0)
    
    print(f"\n Problem file: {problem_file}")
    print(f" Time limit: {time_limit} seconds")
    print(f" Start city: {start_city}")
    
    if not os.path.exists(problem_file):
        print(f"\n Error: Problem file not found: {problem_file}")
        return
    
    try:
        # Load problem
        problem = TSPProblem(problem_file=problem_file)
        print(f"\n Loaded: {problem.name} ({problem.dim} cities)")
        
        # Check problem size
        if problem.dim > 15:
            print(f"\n Warning: Problem size is large ({problem.dim} cities)")
            print(" A* may take a long time or run out of memory.")
            confirm = input(" Continue anyway? (y/n) [n]: ").strip().lower()
            if confirm != 'y':
                print(" Cancelled.")
                return
        
        # Run A*
        print(f"\n Running A* search...")
        solver = TSPAStarSolver(problem, time_limit=time_limit)
        result = solver.solve(start_city=start_city)
        
        print(f"\n" + "="*80)
        print(" A* Search Results")
        print("="*80)
        print(f" Best tour length: {result['cost']:.6f}")
        print(f" Computation time: {result['time_elapsed']:.2f} seconds")
        print(f" Optimal solution: {'Yes' if result['optimal'] else 'No'}")
        
        if result.get('timed_out', False):
            print(f" Status: TIMEOUT (reached time limit)")
        else:
            print(f" Status: COMPLETED")
        
        if result['tour']:
            print(f" Tour: {result['tour']}")
        
        print("="*80)
        
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()


def run_discrete_performance_comparison():
    """Compare ACO vs A* performance."""
    print("\n" + "="*80)
    print(" Performance Comparison - ACO vs A*")
    print("="*80)
    
    print("\n This will run ACO and A* on multiple TSP instances")
    print(" and compare their performance.")
    print("\n Warning: This may take several minutes,")
    print(" depending on problem sizes.")
    
    confirm = input("\n Proceed? (y/n) [n]: ").strip().lower()
    if confirm != 'y':
        print(" Cancelled.")
        return
    
    try:
        # Check if compare config exists
        config_file = 'configs/compare_config.yaml'
        if not os.path.exists(config_file):
            print(f"\n Error: Config file not found: {config_file}")
            print(" Please create this file with TSP problem instances to compare.")
            return
        
        print("\n Running performance comparison...")
        run_tsp_comparison(config_file=config_file)
        
        print("\n Performance comparison completed!")
        print(" Results: results/discrete/performance/")
        
        # Auto-visualize
        print("\n Creating visualizations...")
        visualize_performance_comparison(
            input_dir='results/discrete/performance',
            output_dir='visualizations/discrete/performance'
        )
        
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()


def run_discrete_convergence_visualization():
    """Visualize ACO convergence."""
    print("\n" + "="*80)
    print(" Convergence Visualization - ACO")
    print("="*80)
    
    # Default convergence data file
    input_file = 'results/discrete/convergence/tsp_aco_convergence.csv'
    output_dir = 'visualizations/discrete/convergence'
    
    print(f"\n Input file: {input_file}")
    
    if not os.path.exists(input_file):
        print(f"\n Warning: Convergence data not found.")
        print(" Run ACO optimization first (Option 1).")
        
        run_now = input("\n Run ACO optimization now? (y/n) [y]: ").strip().lower() or 'y'
        if run_now == 'y':
            run_aco_optimization()
            print("\n Now creating convergence plot...")
        else:
            print(" Cancelled.")
            return
    
    try:
        print("\n Creating convergence visualization...")
        plot_convergence_from_file(
            input_file=input_file,
            output_dir=output_dir
        )
        
        print("\n Convergence visualization completed!")
        print(f" Output: {output_dir}/")
        
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()


def run_discrete_sensitivity_analysis():
    """Run sensitivity analysis for ACO."""
    print("\n" + "="*80)
    print(" Sensitivity Analysis - ACO")
    print("="*80)
    
    print("\n This will analyze how ACO parameters affect performance:")
    print("   - n_ants")
    print("   - alpha (pheromone importance)")
    print("   - beta (heuristic importance)")
    print("   - rho (evaporation rate)")
    
    print("\n Warning: This may take 10-15 minutes depending on configuration.")
    
    confirm = input("\n Proceed? (y/n) [n]: ").strip().lower()
    if confirm != 'y':
        print(" Cancelled.")
        return
    
    try:
        print("\n Running sensitivity analysis...")
        visualize_discrete_sensitivity(
            algorithm_name='aco',
            input_dir='results/discrete/parameter_sensitivity',
            output_dir='visualizations/discrete/parameter_sensitivity',
            auto_run=True
        )
        
        print("\n Sensitivity analysis completed!")
        print(f" Results: results/discrete/parameter_sensitivity/aco/")
        print(f" Visualizations: visualizations/discrete/parameter_sensitivity/aco/")
        
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()

