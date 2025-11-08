import numpy as np
class AckleyFunction:
    def __init__(self, dimensions=2, a=20, b=0.2, c=2*np.pi):
        """
        Initialize Ackley funcition params
        Args: 
            dimensions: Number of dimensions
            a, b, c: Ackley funciton parameters
        """
        self.dimensions = dimensions
        self.a = a
        self.b = b
        self.c = c
        self.bounds = (-32.768, 32.768) # Standard bounds for Ackley function (Searching domain, or -32.768 <= x <= 32.768)
        self.global_optimum = np.zeros(dimensions)
        self.global_optimum_value = 0.0

    def evaluate(self, x):
        """
        Evaluate the Ackley funcition at point x
        Args:
            x: np array of shape (dimensions)
        Returns:
            Function value at x
        """
        d = len(x)
        sum_sq = np.sum(x**2)
        sum_cos = np.sum(np.cos(self.c * x))

        term1 = -self.a * np.exp(-self.b * np.sqrt(sum_sq / d))
        term2 = -np.exp(sum_cos / d)
        
        return term1 + term2 + self.a + np.e
        
        # return -self.a * np.exp(-self.b * np.sqrt(1/d * sum_sq)) - np.exp(1/d * sum_cos) + self.a + np.e
    
    def get_bounds(self):
        return [self.bounds] * self.dimensions

    def get_name(self):
        """Return the function name."""
        return f"Ackley Function ({self.dimensions}D)"