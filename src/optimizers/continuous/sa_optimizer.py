import numpy as np
from typing import Callable, List, Tuple

class SimulatedAnnealing:
    """
    Simulated Annealing optimizer.
    """
    
    def __init__(
        self,
        objective_function: Callable,
        bounds: List[Tuple[float, float]],
        max_iterations: int = 100,
        initial_temp: float = 100,
        cooling_rate: float = 0.95,
        neighbor_scale: float = 0.1,
        random_seed: int = None
    ):
        """
        Args:
            objective_function: Function to minimize
            bounds: List of (min, max) tuples for each dimension
            max_iterations: Maximum number of iterations
            initial_temp: Initial temperature
            cooling_rate: Temperature reduction rate (0 < rate < 1)
            neighbor_scale: Scale for neighbor generation
            random_seed: Random seed for reproducibility
        """
        self.objective_function = objective_function
        self.bounds = np.array(bounds)
        self.dimensions = len(bounds)
        self.max_iterations = max_iterations
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.neighbor_scale = neighbor_scale
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # History tracking (giống GA/FA)
        self.best_fitness_history = []
        self.temperature_history = []
        self.current_fitness_history = []
        self.best_solution = None
        self.best_fitness = float('inf')
    
    def initialize_solution(self):
        """Create initial random solution within bounds."""
        solution = np.random.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1],
            size=self.dimensions
        )
        return solution
    
    def generate_neighbor(self, current_solution, temperature):
        """Generate neighbor solution."""
        neighbor = current_solution + np.random.normal(
            0, 1, self.dimensions
        ) * temperature * self.neighbor_scale
        
        # Clip to bounds
        neighbor = np.clip(neighbor, self.bounds[:, 0], self.bounds[:, 1])
        return neighbor
    
    def acceptance_probability(self, delta, temperature):
        """Calculate acceptance probability (Metropolis criterion)."""
        if delta < 0:
            return 1.0
        return np.exp(-delta / temperature)
    
    def optimize(self, verbose=True):
        # Initialize
        current_solution = self.initialize_solution()
        current_fitness = self.objective_function(current_solution)
        
        self.best_solution = current_solution.copy()
        self.best_fitness = current_fitness
        
        temperature = self.initial_temp
        
        for iteration in range(self.max_iterations):
            # Generate neighbor
            neighbor = self.generate_neighbor(current_solution, temperature)
            neighbor_fitness = self.objective_function(neighbor)
            
            # Calculate delta
            delta = neighbor_fitness - current_fitness
            
            # Acceptance criterion
            if np.random.rand() < self.acceptance_probability(delta, temperature):
                current_solution = neighbor
                current_fitness = neighbor_fitness
                
                # Update best
                if current_fitness < self.best_fitness:
                    self.best_solution = current_solution.copy()
                    self.best_fitness = current_fitness
            
            # Track history
            self.best_fitness_history.append(self.best_fitness)
            self.current_fitness_history.append(current_fitness)
            self.temperature_history.append(temperature)
            
            # Cooling
            temperature *= self.cooling_rate
            
            if verbose and (iteration % 10 == 0 or iteration == self.max_iterations - 1):
                print(f"Iteration {iteration}: Best Fitness = {self.best_fitness:.6f}, "
                      f"Current Fitness = {current_fitness:.6f}, Temp = {temperature:.2f}")
        
        return self.best_solution, self.best_fitness
    
    def get_history(self):
        """Return optimization history (giống GA/FA)."""
        return {
            'best_fitness': self.best_fitness_history,
            'current_fitness': self.current_fitness_history,
            'temperature': self.temperature_history
        }