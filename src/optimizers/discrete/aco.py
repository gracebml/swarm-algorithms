import numpy as np
from typing import Dict, Optional, Any
import yaml
import sys
from pathlib import Path

# Ensure src is in path for imports
_current_file = Path(__file__).resolve()
_src_dir = _current_file.parent.parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from problems.discrete.tsp import TSPProblem

"""
Implementation of Max-Min Ant System variation of ACO for solving TSP.
"""

class MMAS:
    def __init__(self, problem: TSPProblem, config: Optional[Dict[str, Any]] = None, **kwargs):
        self.problem = problem

        params = {
            'n_ants': 20,
            'max_iter': 100,
            'alpha': 1.0,
            'beta': 2.0,
            'rho': 0.1,
            'u_gb': 1,
            'stagnation': 50,
        }
    
        if config:
            params.update(config)
            
        params.update(kwargs)
    
        self.config = params
        self.n_ants = int(params['n_ants'])
        self.max_iter = int(params['max_iter'])
        self.alpha = float(params['alpha'])
        self.beta = float(params['beta'])
        self.rho = float(params['rho'])
        self.u_gb = int(params['u_gb'])
        self.stagnation = int(params['stagnation'])
            
        self.dim = problem.get_dim()
        self.distance_matrix = problem.distance_matrix
        self.eta = problem.eta
        if self.n_ants == -1:
            self.n_ants = self.dim
        
        self.best_so_far_tour = None
        self.best_so_far_length = float('inf')
        self.restart_best_tour = None
        self.restart_best_length = float('inf')
        
        Cnn = problem._nearest_neighbor_tour_length()
        
        self.trail_max = 1.0 / (self.rho * Cnn)
        self.trail_min = self.trail_max / (2.0 * self.dim)
        
        self.pheromone = np.full((self.dim, self.dim), self.trail_max)
        
        self.choice_info = self._compute_choice_information()
        
        self.history = []
        self.found_best_iter = 0
        self.restart_found_best_iter = 0

    def optimize(self):
        self.history = []
        iteration_stats = []

        for iteration in range(self.max_iter):
            all_tours, all_lengths = self._construct_solutions()
            
            iter_best_idx = np.argmin(all_lengths)
            iter_best_length = all_lengths[iter_best_idx]
            iter_best_tour = all_tours[iter_best_idx]
            
            stats = self._update_statistics(iteration, iter_best_tour, iter_best_length, all_lengths)
            iteration_stats.append(stats)
            
            self._evaporate()
            self._mmas_pheromone_update(iteration, iter_best_tour, iter_best_length)
            self._check_pheromone_limits()
            self._search_control(iteration)
            self.choice_info = self._compute_choice_information()

        return {
            'best_tour': self.best_so_far_tour,
            'best_fitness': self.best_so_far_length,
            'found_best_iter': self.found_best_iter,
            'history': self.history,
            'iteration_stats': iteration_stats,
            'config': self.config
        }

    def _construct_solutions(self):
        all_tours = []
        all_lengths = []
        
        for _ in range(self.n_ants):
            tour, length = self._build_ant_tour()
            all_tours.append(tour)
            all_lengths.append(length)
            
        return all_tours, all_lengths

    def _build_ant_tour(self):
        tour = []
        visited = np.zeros(self.dim, dtype=bool)
        
        current_city = np.random.randint(0, self.dim)
        tour.append(current_city)
        visited[current_city] = True
        
        while len(tour) < self.dim:
            probs = self._calculate_probabilities(current_city, visited)
            next_city = np.random.choice(self.dim, p=probs)
            tour.append(next_city)
            visited[next_city] = True
            current_city = next_city
            
        tour_length = self._compute_tour_length(tour)
        return tour, tour_length

    def _calculate_probabilities(self, current_city, visited):
        choice_values = self.choice_info[current_city].copy()
        choice_values[visited] = 0.0
        sum_choice = np.sum(choice_values)
        
        if sum_choice == 0.0:
            unvisited = np.where(~visited)[0]
            if len(unvisited) == 0:
                return np.zeros(self.dim)
            probs = np.zeros(self.dim)
            probs[unvisited] = 1.0 / len(unvisited)
        else:
            probs = choice_values / sum_choice
            
        return probs

    def _compute_choice_information(self):
        return self.pheromone ** self.alpha * self.eta ** self.beta

    def _evaporate(self):
        self.pheromone *= (1.0 - self.rho)

    def _mmas_pheromone_update(self, iteration, iter_best_tour, iter_best_length):
        if (iteration % self.u_gb) == 0:
            no_improvement = iteration - self.restart_found_best_iter
            if self.u_gb == 1 and no_improvement > self.stagnation:
                tour_to_update = self.best_so_far_tour
                length_to_update = self.best_so_far_length
            else:
                tour_to_update = self.restart_best_tour
                length_to_update = self.restart_best_length
        else:
            tour_to_update = iter_best_tour
            length_to_update = iter_best_length
        
        if length_to_update > 0 and length_to_update < float('inf'):
            self._deposit_pheromone(tour_to_update, length_to_update)

    def _deposit_pheromone(self, tour, length):
        delta = 1.0 / length
        for i in range(self.dim):
            city_a = tour[i]
            city_b = tour[(i + 1) % self.dim]
            self.pheromone[city_a, city_b] += delta
            self.pheromone[city_b, city_a] += delta

    def _check_pheromone_limits(self):
        np.clip(self.pheromone, self.trail_min, self.trail_max, out=self.pheromone)

    def _update_statistics(self, iteration, iter_best_tour, iter_best_length, all_lengths):
        stats = {
            'iteration': iteration,
            'best_length': iter_best_length,
            'best_so_far_length': self.best_so_far_length,
        }
        
        if iter_best_length < self.best_so_far_length:
            self.best_so_far_tour = iter_best_tour.copy()
            self.best_so_far_length = iter_best_length
            self.found_best_iter = iteration
            self.trail_max = 1.0 / (self.rho * self.best_so_far_length)
            self.trail_min = self.trail_max / (2.0 * self.dim)
        
        if iter_best_length < self.restart_best_length:
            self.restart_best_tour = iter_best_tour.copy()
            self.restart_best_length = iter_best_length
            self.restart_found_best_iter = iteration
        
        self.history.append(self.best_so_far_length)
        return stats

    def _search_control(self, iteration):
        if not (iteration % 100):
            self.restart_best_length = float('inf')
            self.pheromone.fill(self.trail_max)
            self.choice_info = self._compute_choice_information()

    def _compute_tour_length(self, tour):
        length = 0.0
        for i in range(self.dim):
            city_a = tour[i]
            city_b = tour[(i + 1) % self.dim]
            length += self.distance_matrix[city_a, city_b]
        return length


def load_config(config_file: str = '../../../configs/aco_config.yaml'):
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        return {
            'n_ants': 20,
            'max_iter': 100,
            'alpha': 1.0,
            'beta': 2.0,
            'rho': 0.1,
            'u_gb': 1,
            'stagnation': 50
        }
