import numpy as np
from typing import Callable, List, Tuple

class GeneticAlgorithm:
    def __init__(
        self,
        objective_function: Callable,
        bounds: List[Tuple[float, float]],
        population_size: int = 50,
        generations: int = 100,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        elite_size: int = 2,
        tournament_size: int = 3,
        random_seed: int = None
    ):
        """
        Args:
            objective_function: Function to minimize
            bounds: List of (min, max) tuples for each dimension
            population_size: Number of individuals in population
            generations: Number of generations to evolve
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            elite_size: Number of best individuals to preserve
            tournament_size: Size of tournament for selection
            random_seed: Random seed for reproducibility
        """
        self.objective_function = objective_function
        self.bounds = np.array(bounds)
        self.dimensions = len(bounds)
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.tournament_size = tournament_size
        
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
        """Evaluate fitness of all individuals in population."""
        fitness = np.array([self.objective_function(ind) for ind in population])
        return fitness
    
    def tournament_selection(self, population, fitness):
        """Select individual using tournament selection."""
        indices = np.random.choice(len(population), self.tournament_size, replace=False)
        tournament_fitness = fitness[indices]
        winner_idx = indices[np.argmin(tournament_fitness)]
        return population[winner_idx].copy()
    
    def crossover(self, parent1, parent2):
        """Perform uniform crossover between two parents."""
        if np.random.random() < self.crossover_rate:
            mask = np.random.random(self.dimensions) < 0.5
            child1 = np.where(mask, parent1, parent2)
            child2 = np.where(mask, parent2, parent1)
            return child1, child2
        return parent1.copy(), parent2.copy()
    
    def mutate(self, individual):
        """Perform Gaussian mutation on individual."""
        for i in range(self.dimensions):
            if np.random.random() < self.mutation_rate:
                # Gaussian mutation
                mutation_range = self.bounds[i, 1] - self.bounds[i, 0]
                mutation = np.random.normal(0, 0.1 * mutation_range)
                individual[i] += mutation
                # Ensure within bounds
                individual[i] = np.clip(individual[i], self.bounds[i, 0], self.bounds[i, 1])
        return individual
    
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
            
            # Create new population
            new_population = []
            
            # Elitism: keep best individuals
            elite_indices = np.argsort(fitness)[:self.elite_size]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self.tournament_selection(population, fitness)
                parent2 = self.tournament_selection(population, fitness)
                
                # Crossover
                child1, child2 = self.crossover(parent1, parent2)
                
                # Mutation
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            # Update population
            population = np.array(new_population[:self.population_size])
            fitness = self.evaluate_population(population)
        
        return self.best_solution, self.best_fitness
    
    def get_history(self):
        """Return optimization history."""
        return {
            'best_fitness': self.best_fitness_history,
            'mean_fitness': self.mean_fitness_history,
            'population': self.population_history
        }
