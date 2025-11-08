import numpy as np
from scipy import special

class CuckooSearch:

    def __init__(self, n_nests=25, pa=0.25, beta=1.5, alpha=0.01, max_iter=100, seed=None):
        self.n_nests = n_nests
        self.pa = pa
        self.beta = beta
        self.alpha = alpha
        self.max_iter = max_iter
        self.seed = seed

        self.dim = None
        self.bounds = None
        self.lower = None
        self.upper = None
        self.objective_func = None
        self.minimize = True

        # History tracking
        self.best_history = []
        self.mean_history = []
        self.diversity_history = []

    def optimize(self, objective_func, dim, bounds, max_iter=None,
                 minimize=True, verbose=False):
        # Use instance max_iter if not provided
        if max_iter is None:
            max_iter = self.max_iter
        if self.seed is not None:
            np.random.seed(self.seed)

        self.dim = dim
        self.bounds = bounds
        self.lower, self.upper = bounds
        self.objective_func = objective_func
        self.minimize = minimize

        # Initialize nests
        nests = np.random.uniform(self.lower, self.upper, (self.n_nests, self.dim))
        fitness = np.array([objective_func(nest) for nest in nests])

        best_idx = np.argmin(fitness) if minimize else np.argmax(fitness)
        best_nest = nests[best_idx].copy()
        best_fitness = fitness[best_idx]

        # Initialize history
        self.best_history = [best_fitness]
        self.mean_history = [np.mean(fitness)]
        self.diversity_history = [self._calculate_diversity(nests)]

        for iteration in range(max_iter):
            # --- Phase 1: Lévy flights ---
            nests, fitness = self._levy_flight_phase(nests, fitness, best_nest)

            # --- Phase 2: Discovery / Abandon worst nests ---
            nests, fitness = self._discovery_phase(nests, fitness)

            # Update best solution
            current_best_idx = np.argmin(fitness) if minimize else np.argmax(fitness)
            if self._is_better(fitness[current_best_idx], best_fitness):
                best_nest = nests[current_best_idx].copy()
                best_fitness = fitness[current_best_idx]

            # Track history
            self.best_history.append(best_fitness)
            self.mean_history.append(np.mean(fitness))
            self.diversity_history.append(self._calculate_diversity(nests))

            # Log
            if verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{max_iter}: "
                      f"Best = {best_fitness:.6f}, Mean = {self.mean_history[-1]:.6f}")

        return {
            'best_position': best_nest,
            'best_fitness': best_fitness,
            'history': self.best_history,
            'mean_history': self.mean_history,
            'diversity_history': self.diversity_history,
            'nests': nests,
            'fitness': fitness
        }


    def _levy_flight_phase(self, nests, fitness, best_nest):
        new_nests = nests.copy()
        for i in range(self.n_nests):
            levy_step = self._levy_flight()
            step_size = self.alpha * levy_step * (nests[i] - best_nest)
            new_nest = nests[i] + step_size
            new_nest = self._handle_boundaries(new_nest)
            new_fitness = self.objective_func(new_nest)
            j = np.random.randint(self.n_nests)
            if self._is_better(new_fitness, fitness[j]):
                new_nests[j] = new_nest
                fitness[j] = new_fitness
        return new_nests, fitness

    def _discovery_phase(self, nests, fitness):
        n_abandon = int(self.pa * self.n_nests)
        if n_abandon == 0:
            return nests, fitness

        if self.minimize:
            worst_indices = np.argsort(fitness)[-n_abandon:]
        else:
            worst_indices = np.argsort(fitness)[:n_abandon]

        for idx in worst_indices:
            rand_idx1, rand_idx2 = np.random.choice(self.n_nests, 2, replace=False)
            step = np.random.randn(self.dim) * (nests[rand_idx1] - nests[rand_idx2])
            new_nest = nests[idx] + step * self.pa
            new_nest = self._handle_boundaries(new_nest)
            nests[idx] = new_nest
            fitness[idx] = self.objective_func(new_nest)
        return nests, fitness

    def _levy_flight(self):
        numerator = special.gamma(1 + self.beta) * np.sin(np.pi * self.beta / 2)
        denominator = special.gamma((1 + self.beta) / 2) * self.beta * (2 ** ((self.beta - 1) / 2))
        sigma = (numerator / denominator) ** (1 / self.beta)
        u = np.random.normal(0, sigma, self.dim)
        v = np.random.normal(0, 1, self.dim)
        return u / (np.abs(v) ** (1 / self.beta))

    def _handle_boundaries(self, position):
        reflected = position.copy()
        lower_violations = reflected < self.lower
        reflected[lower_violations] = 2 * self.lower - reflected[lower_violations]
        upper_violations = reflected > self.upper
        reflected[upper_violations] = 2 * self.upper - reflected[upper_violations]
        return np.clip(reflected, self.lower, self.upper)

    def _is_better(self, f1, f2):
        return f1 < f2 if self.minimize else f1 > f2

    def _calculate_diversity(self, nests):
        if len(nests) < 2:
            return 0.0
        distances = []
        for i in range(len(nests)):
            for j in range(i + 1, len(nests)):
                distances.append(np.linalg.norm(nests[i] - nests[j]))
        return np.mean(distances)


#Hàm tiện ích 

# def run_cs(objective_func, dim, bounds, n_nests=25, max_iter=100,
#            pa=0.25, beta=1.5, alpha=0.01, minimize=True, seed=None, verbose=False):
#     cs = CuckooSearch(n_nests, pa, beta, alpha, seed)
#     return cs.optimize(objective_func, dim, bounds, max_iter, minimize, verbose)


# #  Hàm mục tiêu 

# def sphere(x): return np.sum(x ** 2)
# def rastrigin(x): return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))
# def rosenbrock(x): return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)
# def ackley(x): return -20 * np.exp(-0.2 * np.sqrt(np.mean(x ** 2))) - np.exp(np.mean(np.cos(2 * np.pi * x))) + 20 + np.e


# #  Hàm main để test 

# def main():
#     print("Test Cuckoo Search (CS) Algorithm")
#     functions = {
#         "Sphere": sphere,
#         "Rastrigin": rastrigin,
#         "Rosenbrock": rosenbrock,
#         "Ackley": ackley
#     }

#     for name, func in functions.items():
#         print(f"\nTesting on {name} function")
#         result = run_cs(
#             objective_func=func,
#             dim=10,
#             bounds=(-5, 5),
#             n_nests=50,
#             max_iter=300,
#             pa=0.3,
#             beta=1.5,
#             alpha=0.2,
#             seed=42,
#             minimize=True,
#             verbose=True
#         )
#         print(f"Best fitness: {result['best_fitness']:.6f}")
#         print(f"Best position (first 5 dims): {result['best_position'][:5]}")

#         plt.plot(result['history'], label=name)

#     plt.title("Cuckoo Search Convergence Curves")
#     plt.xlabel("Iteration")
#     plt.ylabel("Best Fitness")
#     plt.legend()
#     plt.grid(True)
#     plt.show()


# if __name__ == "__main__":
#     main()
