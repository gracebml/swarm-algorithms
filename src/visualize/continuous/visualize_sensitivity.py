import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import sys
from pathlib import Path

# # Setup path to import from src directory
# _current_file = Path(__file__).resolve()
# _src_dir = _current_file.parent.parent
# if str(_src_dir) not in sys.path:
#     sys.path.insert(0, str(_src_dir))

# Configure matplotlib
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12


def plot_parameter_sensitivity(func_name, param_name, algorithm_name='cs',
                               input_dir='results/continuous/parameter_sensitivity',
                                   output_dir='visualizations/continuous/parameter_sensitivity'):
    # Load data
    alg_lower = algorithm_name.lower()
    csv_file = Path(input_dir) / alg_lower / f'{alg_lower}_{func_name}_{param_name}_sensitivity.csv'
    
    if not csv_file.exists():
        print(f"File not found: {csv_file}")
        return
    
    df = pd.read_csv(csv_file)
    
    # Create output directory
    output_path = Path(output_dir) / alg_lower
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Mean fitness with error bars
    ax1.errorbar(df['value'], df['best_mean'], yerr=df['best_std'],
                marker='o', markersize=8, linewidth=2, capsize=5,
                label=f'{param_name.upper()}', color='steelblue')
    ax1.set_xlabel(f'Parameter Value ({param_name})', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Best Fitness (Mean ± Std)', fontsize=14, fontweight='bold')
    ax1.set_title(f'{algorithm_name.upper()}: {param_name.upper()} on {func_name.title()}', 
                 fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Add best value marker
    best_idx = df['best_mean'].idxmin()
    ax1.plot(df.loc[best_idx, 'value'], df.loc[best_idx, 'best_mean'], 
            'r*', markersize=20, label=f'Best: {param_name}={df.loc[best_idx, "value"]:.3f}')
    ax1.legend()
    
    # Plot 2: Min and Max fitness range
    ax2.fill_between(df['value'], df['best_min'], df['best_max'], 
                     alpha=0.3, label='Min-Max Range', color='lightcoral')
    ax2.plot(df['value'], df['best_mean'], 'o-', linewidth=2, 
            markersize=8, label='Mean', color='darkred')
    ax2.set_xlabel(f'Parameter Value ({param_name})', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Best Fitness', fontsize=14, fontweight='bold')
    ax2.set_title(f'Performance Variability: {param_name.upper()} on {func_name.title()}', 
                 fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    ax2.legend()
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_path / f'{alg_lower}_{func_name}_{param_name}_sensitivity.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_file.name}")
    plt.close()


def plot_convergence_comparison(func_name, param_name, algorithm_name='cs',
                                input_dir='results/continuous/parameter_sensitivity',
                                output_dir='visualizations/continuous/parameter_sensitivity'):
    """
    Plot convergence curves comparison for different parameter values.
    
    Args:
        func_name: Function name (sphere, ackley, rastrigin)
        param_name: Parameter name (mutation_rate, w, alpha, etc.)
        algorithm_name: Algorithm name (ga, fa, pso, sa, cs)
        input_dir: Directory containing results
        output_dir: Directory to save visualizations
    """
    # Load convergence data
    alg_lower = algorithm_name.lower()
    json_file = Path(input_dir) / alg_lower / f'{alg_lower}_{func_name}_{param_name}_convergence.json'
    
    if not json_file.exists():
        print(f"File not found: {json_file}")
        return
    
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Create output directory
    output_path = Path(output_dir) / alg_lower
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Use color palette
    colors = plt.cm.viridis(np.linspace(0, 1, len(data['parameter_values'])))
    
    # Plot each convergence curve
    for i, (value, curve) in enumerate(zip(data['parameter_values'], data['convergence_curves'])):
        ax.plot(curve, linewidth=2.5, alpha=0.8, color=colors[i],
               label=f'{param_name}={value}')
    
    ax.set_xlabel('Iteration', fontsize=14, fontweight='bold')
    ax.set_ylabel('Best Fitness', fontsize=14, fontweight='bold')
    ax.set_title(f'{algorithm_name.upper()} Convergence: {param_name.upper()} on {func_name.title()}', 
                fontsize=16, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', ncol=2, framealpha=0.9)
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_path / f'{alg_lower}_{func_name}_{param_name}_convergence.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file.name}")
    plt.close()


def plot_heatmap_sensitivity(func_name, algorithm_name='cs',
                             input_dir='results/continuous/parameter_sensitivity',
                             output_dir='visualizations/continuous/parameter_sensitivity'):
    """
    Plot heatmap showing sensitivity of all parameters for an algorithm.
    
    Args:
        func_name: Function name (sphere, ackley, rastrigin)
        algorithm_name: Algorithm name (ga, fa, pso, sa, cs)
        input_dir: Directory containing results
        output_dir: Directory to save visualizations
    """
    alg_lower = algorithm_name.lower()
    alg_path = Path(input_dir) / alg_lower
    
    # Find all parameter CSV files for this function
    all_data = []
    params = []
    
    for csv_file in alg_path.glob(f'{alg_lower}_{func_name}_*_sensitivity.csv'):
        # Extract parameter name from filename
        parts = csv_file.stem.split('_')
        if len(parts) >= 3:
            param = '_'.join(parts[2:-1])  # Handle multi-word parameters
            params.append(param)
            
            df = pd.read_csv(csv_file)
            df['normalized_fitness'] = (df['best_mean'] - df['best_mean'].min()) / \
                                      (df['best_mean'].max() - df['best_mean'].min() + 1e-10)
            all_data.append(df)
    
    if not all_data:
        print(f"No sensitivity data found for {algorithm_name.upper()} on {func_name}")
        return
    
    # Create output directory
    output_path = Path(output_dir) / alg_lower
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create figure
    fig, axes = plt.subplots(1, len(all_data), figsize=(6 * len(all_data), 6))
    
    if len(all_data) == 1:
        axes = [axes]
    
    for idx, (df, param) in enumerate(zip(all_data, params)):
        # Create bar plot
        ax = axes[idx]
        bars = ax.bar(range(len(df)), df['best_mean'], 
                     color=plt.cm.RdYlGn_r(df['normalized_fitness']),
                     edgecolor='black', linewidth=1.5)
        
        ax.set_xlabel(f'{param.upper()} Value', fontsize=12, fontweight='bold')
        ax.set_ylabel('Best Fitness (Mean)', fontsize=12, fontweight='bold')
        ax.set_title(f'{param.upper()} Sensitivity', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels([f'{v:.3f}' for v in df['value']], rotation=45, ha='right')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, val in zip(bars, df['best_mean']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2e}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle(f'{algorithm_name.upper()} Parameter Sensitivity: {func_name.title()} Function', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    output_file = output_path / f'{alg_lower}_{func_name}_all_params_sensitivity.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file.name}")
    plt.close()


def create_summary_table(algorithm_name, func_name,
                        input_dir='results/continuous/parameter_sensitivity',
                        output_dir='results/tables'):
    """
    Create summary table of best parameter values.
    
    Args:
        algorithm_name: Algorithm name (ga, fa, pso, sa, cs)
        func_name: Function name (sphere, ackley, rastrigin)
        input_dir: Directory containing results
        output_dir: Directory to save tables
    """
    alg_lower = algorithm_name.lower()
    alg_path = Path(input_dir) / alg_lower
    
    summary_data = []
    
    # Find all parameter CSV files for this function
    for csv_file in alg_path.glob(f'{alg_lower}_{func_name}_*_sensitivity.csv'):
        # Extract parameter name from filename
        parts = csv_file.stem.split('_')
        if len(parts) >= 3:
            param = '_'.join(parts[2:-1])
            
            df = pd.read_csv(csv_file)
            best_idx = df['best_mean'].idxmin()
            
            summary_data.append({
                'Parameter': param.upper(),
                'Best Value': df.loc[best_idx, 'value'],
                'Best Mean Fitness': df.loc[best_idx, 'best_mean'],
                'Std Dev': df.loc[best_idx, 'best_std'],
                'Min Fitness': df.loc[best_idx, 'best_min'],
                'Max Fitness': df.loc[best_idx, 'best_max']
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        output_file = output_path / f'{alg_lower}_{func_name}_best_parameters.csv'
        summary_df.to_csv(output_file, index=False)
        print(f"Saved summary: {output_file.name}")


def check_and_run_sensitivity_if_needed(algorithm_name, input_dir='results/continuous/parameter_sensitivity'):
    """
    Check if sensitivity results exist, if not run the analysis.
    
    Args:
        algorithm_name: Name of algorithm (ga, fa, pso, sa, cs)
        input_dir: Directory containing results
    
    Returns:
        True if results exist or were successfully generated, False otherwise
    """
    input_path = Path(input_dir) / algorithm_name.lower()
    
    # Check if results directory exists and has CSV files
    if input_path.exists():
        csv_files = list(input_path.glob('*.csv'))
        if csv_files:
            print(f"Found {len(csv_files)} existing result files for {algorithm_name.upper()}")
            return True
    
    # Results don't exist, need to run sensitivity analysis
    print(f"\nNo results found for {algorithm_name.upper()}")
    print(f"Running sensitivity analysis first (this may take a while)...\n")
    
    config_file = Path(f'configs/sensitivity/{algorithm_name.lower()}_sensitivity.yaml')
    
    if not config_file.exists():
        print(f"Error: Config file not found: {config_file}")
        return False
    
    try:
        # Import here to avoid circular dependency
        from src.utils.run_sensitivity_continuous import run_sensitivity_from_config
        
        run_sensitivity_from_config(str(config_file))
        print(f"\nSensitivity analysis complete for {algorithm_name.upper()}")
        return True
    except Exception as e:
        print(f"Error running sensitivity analysis: {e}")
        import traceback
        traceback.print_exc()
        return False


def visualize_algorithm_results(algorithm_name, input_dir='results/continuous/parameter_sensitivity',
                                output_dir='visualizations/continuous/parameter_sensitivity',
                                auto_run=True):
    print(f"\n{'='*70}")
    print(f"Visualizing results for: {algorithm_name.upper()}")
    print(f"{'='*70}")
    
    # Check if results exist, run analysis if needed
    if auto_run:
        if not check_and_run_sensitivity_if_needed(algorithm_name, input_dir):
            print(f"Skipping visualization for {algorithm_name.upper()}")
            return
    
    input_path = Path(input_dir) / algorithm_name.lower()
    
    if not input_path.exists():
        print(f"No results found for {algorithm_name} at {input_path}")
        return
    
    # Find all CSV files
    csv_files = list(input_path.glob('*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {input_path}")
        return
    
    print(f"Found {len(csv_files)} result files\n")
    
    # Extract unique problems and parameters
    problems = set()
    parameters = set()
    
    alg_lower = algorithm_name.lower()
    for csv_file in csv_files:
        parts = csv_file.stem.split('_')
        if len(parts) >= 3 and parts[-1] == 'sensitivity':
            # Format: algorithm_problem_parameter_sensitivity
            problem = parts[1]
            parameter = '_'.join(parts[2:-1])  # Handle multi-word parameters
            problems.add(problem)
            parameters.add(parameter)
    
    print(f"Problems: {sorted(problems)}")
    print(f"Parameters: {sorted(parameters)}\n")
    
    # Create visualizations
    for problem in sorted(problems):
        print(f"\n{problem.upper()} Function:")
        for parameter in sorted(parameters):
            try:
                # Parameter sensitivity plot
                plot_parameter_sensitivity(
                    func_name=problem,
                    param_name=parameter,
                    algorithm_name=algorithm_name,
                    input_dir=input_dir,
                    output_dir=output_dir
                )
                
                # Convergence comparison plot
                plot_convergence_comparison(
                    func_name=problem,
                    param_name=parameter,
                    algorithm_name=algorithm_name,
                    input_dir=input_dir,
                    output_dir=output_dir
                )
                
            except Exception as e:
                print(f"Warning: Could not create plot for {parameter} - {e}")
        
        # Combined heatmap for this problem
        try:
            plot_heatmap_sensitivity(
                func_name=problem,
                algorithm_name=algorithm_name,
                input_dir=input_dir,
                output_dir=output_dir
            )
        except Exception as e:
            print(f"Warning: Could not create heatmap - {e}")
        
        # Summary table for this problem
        try:
            create_summary_table(
                algorithm_name=algorithm_name,
                func_name=problem,
                input_dir=input_dir,
                output_dir='results/tables'
            )
        except Exception as e:
            print(f"Warning: Could not create summary table - {e}")
    
    print(f"\nVisualization complete for {algorithm_name.upper()}")


def visualize_all_results(input_dir='results/continuous/parameter_sensitivity',
                          output_dir='visualizations/continuous/parameter_sensitivity',
                          auto_run=True):
    algorithms = ['fa', 'pso', 'cs', 'abc']
    
    for algorithm in algorithms:
        visualize_algorithm_results(algorithm, input_dir, output_dir, auto_run=auto_run)
