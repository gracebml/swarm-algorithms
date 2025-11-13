import numpy as np

class RastriginFunction:
    def __init__(self, dimensions=2, A=10):
        """
        Args: 
            dimensions: Number of dimensions
            A: Rastrigin function parameter
        """
        self.dimensions = dimensions
        self.A = A
        self.bounds = (-5.12, 5.12)  # Standard bounds for Rastrigin function
        self.global_optimum = np.zeros(dimensions)
        self.global_optimum_value = 0.0

    def evaluate(self, x):
        """
        Evaluate the Rastrigin function at point x
        Args:
            x: np array of shape (dimensions)
        Returns:
            Function value at x
        """
        d = len(x)
        sum_term = np.sum(x**2 - self.A * np.cos(2 * np.pi * x))
        
        return self.A * d + sum_term
    
    def get_bounds(self):
        return [self.bounds] * self.dimensions

    def get_name(self):
        """Return the function name."""
        return f"Rastrigin Function ({self.dimensions}D)"

