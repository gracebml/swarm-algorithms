"""
3D Landscape Visualization for Optimization Functions
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


def sphere(x):
    """Sphere function"""
    x = np.asarray(x)
    return np.sum(x ** 2)


def ackley(x):
    """Ackley function"""
    x = np.asarray(x)
    return -20 * np.exp(-0.2 * np.sqrt(np.mean(x ** 2))) - \
           np.exp(np.mean(np.cos(2 * np.pi * x))) + 20 + np.e


def rastrigin(x):
    """Rastrigin function"""
    x = np.asarray(x)
    return 10 * len(x) + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


def rosenbrock(x):
    """Rosenbrock function"""
    x = np.asarray(x)
    return np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


FUNCTIONS = {
    'sphere': sphere,
    'ackley': ackley,
    'rastrigin': rastrigin,
    'rosenbrock': rosenbrock
}


def create_mesh_grid(bounds, resolution=100):
    lower, upper = bounds
    x = np.linspace(lower, upper, resolution)
    y = np.linspace(lower, upper, resolution)
    X, Y = np.meshgrid(x, y)
    return X, Y


def evaluate_function_on_grid(func, X, Y):
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = func([X[i, j], Y[i, j]])
    return Z


def plot_3d_surface(func_name, bounds=(-5, 5), resolution=100, 
                   output_dir='visualizations/continuous/landscapes'):
    """
    Create comprehensive 3D landscape visualization for a given function.
    
    Args:
        func_name: Name of function ('sphere', 'ackley', 'rastrigin')
        bounds: Tuple of (lower, upper) bounds for visualization
        resolution: Number of points in each dimension
        output_dir: Directory to save output
    """
    func = FUNCTIONS[func_name]
    
    # Create mesh grid
    X, Y = create_mesh_grid(bounds, resolution)
    Z = evaluate_function_on_grid(func, X, Y)
    
    # Create figure with 4 subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 3D surface plot
    ax1 = fig.add_subplot(221, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.8,
                           linewidth=0, antialiased=True)
    ax1.set_xlabel('X1', fontsize=12, fontweight='bold')
    ax1.set_ylabel('X2', fontsize=12, fontweight='bold')
    ax1.set_zlabel('f(X1, X2)', fontsize=12, fontweight='bold')
    ax1.set_title(f'{func_name.title()} Function - 3D Surface', 
                 fontsize=14, fontweight='bold')
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)
    
    # Wireframe plot
    ax2 = fig.add_subplot(222, projection='3d')
    ax2.plot_wireframe(X, Y, Z, color='steelblue', alpha=0.6, linewidth=0.5)
    ax2.set_xlabel('X1', fontsize=12, fontweight='bold')
    ax2.set_ylabel('X2', fontsize=12, fontweight='bold')
    ax2.set_zlabel('f(X1, X2)', fontsize=12, fontweight='bold')
    ax2.set_title(f'{func_name.title()} Function - Wireframe', 
                 fontsize=14, fontweight='bold')
    
    # Contour plot (filled)
    ax3 = fig.add_subplot(223)
    contourf = ax3.contourf(X, Y, Z, levels=50, cmap=cm.viridis, alpha=0.8)
    ax3.set_xlabel('X1', fontsize=12, fontweight='bold')
    ax3.set_ylabel('X2', fontsize=12, fontweight='bold')
    ax3.set_title(f'{func_name.title()} Function - Contour (Filled)', 
                 fontsize=14, fontweight='bold')
    fig.colorbar(contourf, ax=ax3)
    ax3.grid(True, alpha=0.3)
    
    # Contour plot (lines)
    ax4 = fig.add_subplot(224)
    contour = ax4.contour(X, Y, Z, levels=30, cmap=cm.viridis, linewidths=1.5)
    ax4.clabel(contour, inline=True, fontsize=8)
    ax4.set_xlabel('X1', fontsize=12, fontweight='bold')
    ax4.set_ylabel('X2', fontsize=12, fontweight='bold')
    ax4.set_title(f'{func_name.title()} Function - Contour (Lines)', 
                 fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'Landscape Visualization: {func_name.title()} Function', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f'{func_name}_landscape.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f" Saved landscape: {output_file}")
    plt.close()


def generate_all_landscapes(functions=['sphere', 'ackley', 'rastrigin'], 
                           bounds=(-5, 5), resolution=100,
                           output_dir='visualizations/continuous/landscapes'):
    """
    Generate 3D landscape visualizations for all specified functions.
    
    Args:
        functions: List of function names to visualize
        bounds: Tuple of (lower, upper) bounds
        resolution: Grid resolution for plotting
        output_dir: Directory to save outputs
    """
    print("\n" + "="*80)
    print(" GENERATING 3D LANDSCAPE VISUALIZATIONS")
    print("="*80)
    
    for func_name in functions:
        print(f"\n Processing {func_name.upper()} function...")
        plot_3d_surface(func_name, bounds, resolution, output_dir)
    
    print("\n" + "="*80)
    print(f" All landscapes saved to: {output_dir}")
    print("="*80)
    