import numpy as np 

class ArtificialBeeColony:
    def __init__(self, objective_function, colony_size=50, max_iterations=1000, limit=None, verbose=True):
        """
        objective_function: Function needs optimization
        colony_size: Total number of bees
        max_iteration
        """
        self.objective_function = objective_function
        self.colony_size = colony_size
        self.num_employed_bees = colony_size // 2
        self.num_onlooker_bees = colony_size // 2
        self.max_iterations = max_iterations
        self.verbose = verbose

        # Get dimensions and bounds
        self.bounds = objective_function.get_bounds()
        self.dimensions = len(self.bounds)

        # Set limit of abandonment
        if limit is None:
            self.limit = self.num_employed_bees * self.dimensions #(default)
        else:
            self.limit = limit 
        
        # init population
        self.food_sources = None
        self.fitness = None # giá trị ban đầu của hàm fitness
        self.trial_counters = None

        # best solution tracking
        self.best_solution = None
        self.best_fitness = None
        self.best_value = float('inf')

        # History for virsualization
        self.history = {
            'best_value' : [],
            'mean_value' : [],
            'food_sources_history' : []
        }

    def initialize_population(self):
        self.food_sources = np.zeros((self.num_employed_bees, self.dimensions))

        for i in range(self.num_employed_bees):
            for j in range(self.dimensions):
                lower, upper = self.bounds[j]
                self.food_sources[i, j] = np.random.uniform(lower, upper)
            
        # Evaluate initial fitness
        self.fitness = np.array([self._calculate_fitness(self.food_sources[i]) for i in range(self.num_employed_bees)])

        # initialize trial counters
        self.trial_counters = np.zeros(self.num_employed_bees)

        # find initial best
        self._update_best_solution()
    
    def _calculate_fitness(self, solution):
        obj_value = self.objective_function.evaluate(solution)

        if obj_value >= 0:
            fitness = 1.0 / (1.0 + obj_value)
        else:
            fitness = 1.0 / (1.0 + abs(obj_value))

        return fitness
    
    def _update_best_solution(self):
        # update nhu the nao
        # duyet qua tat ca food sources
        # cap nhat
        for i in range(self.num_employed_bees):
            obj_value = self.objective_function.evaluate(self.food_sources[i]) # obj_value in R^d
            if obj_value < self.best_value:
                self.best_value = obj_value
                self.best_solution = self.food_sources[i]
                self.best_fitness = self.fitness[i]
        
    def _generate_new_solution(self, current_solution, solution_idx):
        """
        current_solution: Current food source
        solution_idx: Index of current solution
        returns: new candidate solution (from neighbour)
        Formula: v_ij = x_ij + phi_ij * (x_ij - x_kj) , v_ij : new position derives from x_ij
        """
        new_solution = current_solution.copy()
        # randomly select a dimension to modify 
        j = np.random.randint(0, self.dimensions)
        k = solution_idx
        while k == solution_idx:
            k = np.random.randint(0, self.num_employed_bees) # ensure k != i
        
        phi = np.random.uniform(-1, 1)
        
        # modify the selected dimension
        new_solution[j] = current_solution[j] + phi * (current_solution[j] - self.food_sources[k, j])

        # Apply bounds to fit new solution
        lower, upper = self.bounds[j]
        new_solution[j] = np.clip(new_solution[j], lower, upper)

        return new_solution

    def employed_bees_phase(self):
        for i in range(self.num_employed_bees):
            new_solution = self._generate_new_solution(self.food_sources[i], i)
            fitness = self._calculate_fitness(new_solution)
            if self.fitness[i] < fitness:
                self.food_sources[i] = new_solution
                self.fitness[i] = fitness
                self.trial_counters[i] = 0
            else:
                self.trial_counters[i] += 1 # (?)
    
    def calculate_selection_probability(self):
        """
        Calculate selection probabilities for onlooker bees
        """
        total_fitness = np.sum(self.fitness)
        probabilities = self.fitness / total_fitness
        return probabilities
    
    def onlooker_bees_phase(self):
        prob = self.calculate_selection_probability()
        for i in range(self.num_onlooker_bees):
            # select a food source based on probability
            i = np.random.choice(self.num_employed_bees, p=prob)

            # go to i and explore neighbour
            new_solution = self._generate_new_solution(self.food_sources[i], i)
            fitness = self._calculate_fitness(new_solution)

            # greedy selection
            if self.fitness[i] < fitness:
                self.food_sources[i] = new_solution
                self.fitness[i] = fitness
                self.trial_counters[i] = 0
            else:
                self.trial_counters[i] += 1 # tracking failure
    
    def scout_bees_phase(self):
        for i in range(self.num_employed_bees):
            if self.trial_counters[i] >= self.limit: # depleted source => abandon this source
            
                # random initialize new food sources
                for j in range(self.dimensions):
                    lower, upper = self.bounds[j]
                    self.food_sources[i, j] = np.random.uniform(lower, upper)

                # evaluate new food source
                self.fitness[i] = self._calculate_fitness(self.food_sources[i])
                self.trial_counters[i] = 0

    def optimize(self):
        """
        Main optimize loop
        returns:
            - best_solution: vector x in R^d that represent best food source found
            - best_value: f(best_solution), f is objective function
            - history: history of optimization process
        """
        self.initialize_population()

        if self.verbose:
            print('=' * 50)
            print("Artificial Bee Colony Optimization")
            print('=' * 50)
            print(f'Problem Dimensions: {self.dimensions}')
            print(f'Colony Size: {self.colony_size}')
            print(f'Employed Bees: {self.num_employed_bees}')
            print(f'Onlooker Bees: {self.num_onlooker_bees}')
            print(f'Abandon Limit: {self.limit}')
            print(f"Max Iterations: {self.max_iterations}")
            print("=" * 50)
            print(f"{'Iteration':<12} {'Best Value':<15} {'Mean Value':<15} {'Std Dev':<15}")
            print("-" * 50)

        # Main loop
        for iter in range(self.max_iterations):
            # Employed bees phase
            self.employed_bees_phase()

            # Onlooker bees phase
            self.onlooker_bees_phase()

            # Scout bee phase 
            self.scout_bees_phase()

            # Update best solution
            self._update_best_solution()
            
            # Calculate statistics
            current_value = [self.objective_function.evaluate(self.food_sources[i]) for i in range(self.num_employed_bees)]
            mean_val = np.mean(current_value)
            std_val = np.std(current_value)

            # Save history
            self.history['best_value'].append(self.best_value)
            self.history['mean_value'].append(mean_val)
            self.history['food_sources_history'].append(self.food_sources.copy())

            # Print progress
            if self.verbose and (iter % 50 == 0 or iter == self.max_iterations - 1):
                print(f"{iter:<12} {self.best_value:<15.6f} {mean_val:<15.6f} {std_val:<15.6f}")
        
        # Print final results
        if self.verbose:
            print("=" * 50)
            print(f"Optimization Complete!")
            print(f"Best Solution Found: {self.best_solution}")
            print(f"Best Objective Value: {self.best_value:.10f}")
            print(f"Global Optimum: {self.objective_function.global_optimum}")
            print(f"Global Optimum Value: {self.objective_function.global_optimum_value}")
            print(f"Error: {abs(self.best_value - self.objective_function.global_optimum_value):.10e}")
            print("=" * 50)
        
        return {
            'best_solution': self.best_solution,
            'best_value': self.best_value,
            'history': self.history
        }
















        

            


        








        




