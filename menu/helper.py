import os
import sys
from pathlib import Path

# Try to import config loader
try:
    from src.utils.config_loader import load_algorithm_params
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False


def get_algorithm_params(use_yaml=True):
    """Get algorithm parameters from YAML configs or use hardcoded defaults."""
    # Hardcoded defaults (fallback)
    default_params = {
        'GA': {'population_size': 50, 'generations': 100, 'crossover_rate': 0.8, 'mutation_rate': 0.1, 'elite_size': 2, 'tournament_size': 3},
        'FA': {'population_size': 50, 'generations': 100, 'alpha': 0.5, 'beta0': 1.0, 'gamma': 1.0},
        'PSO': {'n': 50, 'max_iter': 100, 'w': 0.95, 'c1': 1.49445, 'c2': 1.49445},
        'SA': {'max_iterations': 100, 'initial_temp': 100, 'cooling_rate': 0.95, 'neighbor_scale': 0.1},
        'CS': {'n_nests': 50, 'max_iter': 100, 'pa': 0.25, 'beta': 1.5, 'alpha': 0.01},
        'ABC': {'colony_size': 50, 'max_iterations': 100, 'limit': 20}
    }
    
    # If explicitly asked to use hardcoded
    if not use_yaml:
        return default_params
    
    # load params from YAML
    if CONFIG_AVAILABLE:
        try:
            print("\n Loading algorithm parameters from YAML configs...")
            algorithms = {
                'GA': load_algorithm_params('ga'),
                'FA': load_algorithm_params('fa'),
                'PSO': load_algorithm_params('pso'),
                'SA': load_algorithm_params('sa'),
                'CS': load_algorithm_params('cs'),
                'ABC': load_algorithm_params('abc')
            }
            
            # Check if any config is empty, fallback to defaults for that algorithm
            all_empty = True
            for alg_name, params in algorithms.items():
                if params:
                    all_empty = False
                else:
                    print(f"   Warning: {alg_name} config empty, using hardcoded defaults")
                    algorithms[alg_name] = default_params[alg_name]
            
            if all_empty:
                print("   Warning: All YAML configs are empty, using hardcoded defaults\n")
                return default_params
            
            print(" YAML configs loaded successfully!\n")
            return algorithms
        except Exception as e:
            print(f"\n Warning: Failed to load YAML configs: {e}")
            print(" Fallback: Using hardcoded defaults...\n")
            return default_params
    else:
        print("\n Warning: config_loader not available.")
        print(" Fallback: Using hardcoded defaults...\n")
        return default_params


def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """Print project header."""
    print("\n" + "="*80)
    print("   AI FUNDAMENTALS - LAB 1: OPTIMIZATION ALGORITHMS")
    print("="*80)
    print("\n   CONTINUOUS PROBLEMS:")
    print("     Swarm-based: PSO, FA, CS, ABC")
    print("     Classical: GA, SA")
    print("     Functions: Sphere, Ackley, Rastrigin")
    print("\n   DISCRETE PROBLEMS:")
    print("     Swarm-based: ACO")
    print("     Classical: A*")
    print("     Problem: TSP (Traveling Salesman)")
    print("\n" + "="*80 + "\n")


def print_menu():
    """Print main menu."""
    print("\n MAIN MENU")
    print("-" * 80)
    print("  1. CONTINUOUS PROBLEMS - Sphere, Ackley, Rastrigin")
    print("  2. DISCRETE PROBLEMS - TSP (Traveling Salesman)")
    print("  3. VIEW RESULTS - Show results directories")
    print("  0. EXIT")
    print("-" * 80)