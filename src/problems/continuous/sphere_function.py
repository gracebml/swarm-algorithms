import numpy as np

class SphereFunction:
    """
    Sphere function for optimization testing.
    
    The Sphere function is a simple convex function with a single global minimum.
    
    Formula:
    f(x) = sum(x_i^2)
    """
    
    def __init__(self, dimensions=2):
        """
        Initialize Sphere function.
        
        Args:
            dimensions: Number of dimensions
        """
        self.dimensions = dimensions
        self.bounds = [(-100, 100)] * dimensions
        self.global_optimum = np.zeros(dimensions)
        self.global_minimum = 0.0
        
    def evaluate(self, x):
        """
        Evaluate the Sphere function at point x.
        
        Args:
            x: Input vector (numpy array or list)
            
        Returns:
            Function value at x
        """
        x = np.array(x)
        return np.sum(x**2)
    
    def get_bounds(self):
        """Return the bounds for each dimension."""
        return self.bounds
    
    def get_name(self):
        """Return the function name."""
        return f"Sphere Function ({self.dimensions}D)"