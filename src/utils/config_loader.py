import yaml
from pathlib import Path


def load_algorithm_config(algorithm_name, config_dir='configs/algorithms'):

    config_file = Path(config_dir) / f'{algorithm_name.lower()}.yaml'
    
    if not config_file.exists():
        return None
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        if config and 'parameters' in config:
            return config['parameters']
        
        return None
    except Exception as e:
        print(f"Warning: Could not load config from {config_file}: {e}")
        return None


def load_problem_config(problem_name, config_dir='configs/problems'):

    config_file = Path(config_dir) / f'problem_{problem_name.lower()}.yaml'
    
    if not config_file.exists():
        return None
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        if config and problem_name.lower() in config:
            return config[problem_name.lower()]
        
        return None
    except Exception as e:
        print(f"Warning: Could not load config from {config_file}: {e}")
        return None


def get_default_algorithm_params(algorithm_name):

    defaults = {
        'GA': {
            'population_size': 50,
            'generations': 100,
            'crossover_rate': 0.8,
            'mutation_rate': 0.1,
            'elite_size': 2,
            'tournament_size': 3
        },
        'FA': {
            'population_size': 50,
            'generations': 100,
            'alpha': 0.5,
            'beta0': 1.0,
            'gamma': 1.0
        },
        'PSO': {
            'n': 50,
            'max_iter': 100,
            'w': 0.95,
            'c1': 1.49445,
            'c2': 1.49445
        },
        'SA': {
            'max_iterations': 100,
            'initial_temp': 100,
            'cooling_rate': 0.95,
            'neighbor_scale': 0.1
        },
        'CS': {
            'n_nests': 50,
            'max_iter': 100,
            'pa': 0.25,
            'beta': 1.5,
            'alpha': 0.01
        },
        'ABC': {
            'colony_size': 50,
            'max_iterations': 100,
            'limit': 20
        }
    }
    
    return defaults.get(algorithm_name.upper(), {})


def load_algorithm_params(algorithm_name, prefer_yaml=True):
    """
    Load algorithm parameters from YAML or use defaults.
    
    Args:
        algorithm_name: Name of algorithm
        prefer_yaml: If True, try YAML first, then fallback to defaults
    
    Returns:
        Dictionary with algorithm parameters
    """
    if prefer_yaml:
        params = load_algorithm_config(algorithm_name)
        if params is not None:
            return params
    
    # Fallback to defaults
    return get_default_algorithm_params(algorithm_name)

