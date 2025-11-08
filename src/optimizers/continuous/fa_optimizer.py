import numpy as np
from typing import Callable, List, Tuple


class FireflyAlgorithm:
    def __init__(
        self,
        objective_function: Callable,
        bounds: List[Tuple[float, float]],
        population_size: int = 50,
        generations: int = 100,
        alpha: float = 0.5,
        beta0: float = 1.0,
        gamma: float = 1.0,
        random_seed: int = None
    ):
        """
        Initialize Firefly Algorithm.
        
        Args:
            objective_function: Function to minimize
            bounds: List of (min, max) tuples for each dimension
            population_size: Number of fireflies
            generations: Number of iterations
            alpha: Randomization parameter (step size)
            beta0: Attractiveness at distance 0
            gamma: Light absorption coefficient
            random_seed: Random seed for reproducibility
        """
        self.objective_function = objective_function
        self.bounds = np.array(bounds)
        self.dimensions = len(bounds)
        self.population_size = population_size
        self.generations = generations
        self.alpha = alpha
        self.alpha_initial = alpha
        self.beta0 = beta0
        self.gamma = gamma
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # History tracking
        self.best_fitness_history = []
        self.mean_fitness_history = []
        self.population_history = []
        self.best_solution = None
        self.best_fitness = float('inf')
        
    def initialize_population(self):
        """Create initial random population within bounds."""
        population = np.random.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1],
            size=(self.population_size, self.dimensions)
        )
        return population
    
    def evaluate_population(self, population):
        """Evaluate fitness (light intensity) of all fireflies."""
        fitness = np.array([self.objective_function(ind) for ind in population])
        return fitness
    
    def distance(self, firefly_i, firefly_j):
        """Calculate Euclidean distance between two fireflies."""
        return np.sqrt(np.sum((firefly_i - firefly_j)**2))
    
    def attractiveness(self, distance):
        """Calculate attractiveness based on distance."""
        return self.beta0 * np.exp(-self.gamma * distance**2)
    
    def move_firefly(self, firefly_i, firefly_j, beta):
        """Move firefly i towards firefly j."""
        # Attraction term
        attraction = beta * (firefly_j - firefly_i)
        
        # Random term
        random_term = self.alpha * (np.random.random(self.dimensions) - 0.5)
        
        # Update position
        new_position = firefly_i + attraction + random_term
        
        # Ensure within bounds
        new_position = np.clip(new_position, self.bounds[:, 0], self.bounds[:, 1])
        
        return new_position
    
    def optimize(self, verbose=True):
        # Initialize population
        population = self.initialize_population()
        fitness = self.evaluate_population(population)
        
        for generation in range(self.generations):
            # Track best and mean fitness
            best_idx = np.argmin(fitness)
            self.best_fitness_history.append(fitness[best_idx])
            self.mean_fitness_history.append(np.mean(fitness))
            self.population_history.append(population.copy())
            
            if fitness[best_idx] < self.best_fitness:
                self.best_fitness = fitness[best_idx]
                self.best_solution = population[best_idx].copy()
            
            if verbose and (generation % 10 == 0 or generation == self.generations - 1):
                print(f"Generation {generation}: Best Fitness = {fitness[best_idx]:.6f}, "
                      f"Mean Fitness = {np.mean(fitness):.6f}")
            
            # Move fireflies
            new_population = population.copy()
            
            for i in range(self.population_size):
                for j in range(self.population_size):
                    # If firefly j is brighter (better fitness = lower value)
                    if fitness[j] < fitness[i]:
                        # Calculate distance
                        r = self.distance(population[i], population[j])
                        
                        # Calculate attractiveness
                        beta = self.attractiveness(r)
                        
                        # Move firefly i towards j
                        new_population[i] = self.move_firefly(
                            population[i], 
                            population[j], 
                            beta
                        )
                        
                        # Update fitness
                        fitness[i] = self.objective_function(new_population[i])
            
            population = new_population
            
            # Reduce randomization parameter (cooling)
            self.alpha = self.alpha_initial * (0.95 ** generation)
        
        return self.best_solution, self.best_fitness
    
    def get_history(self):
        """Return optimization history."""
        return {
            'best_fitness': self.best_fitness_history,
            'mean_fitness': self.mean_fitness_history,
            'population': self.population_history
        }


# class FireflyAlgorithm:
#     def __init__(self):
#         pass

#     def initialize_population():
#         pass

#     def _calculate_light_intensity():
#         pass
    
#     def update_firefly_positions():
#         pass
#         # based on attractiveness
    
    



    