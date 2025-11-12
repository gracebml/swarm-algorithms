"""
Continuous optimization problems menu and functions (Sphere, Ackley, Rastrigin)
"""

import sys
import os
from pathlib import Path
import pandas as pd

# Path setup is done by main.py, so imports work when running via main.py

# Continuous visualizations
from src.visualize.continuous.visualize_convergence import visualize_all_convergence
from src.visualize.continuous.visualize_overview_performance import visualize_all_comparative
from src.visualize.continuous.visualize_swarm_classic_comparison import visualize_all_paradigm_comparison
from src.visualize.continuous.visualize_swarm_only_performance import visualize_all_swarm_metrics
from src.visualize.continuous.visualize_sensitivity import visualize_algorithm_results, visualize_all_results
from src.visualize.continuous.visualize_landscape import plot_3d_surface, generate_all_landscapes

from src.utils.compare_metrics_continuous import benchmark_algorithms, benchmark_scalability
from src.problems.continuous.sphere_function import SphereFunction
from src.problems.continuous.ackley_function import AckleyFunction
from src.problems.continuous.rastrigin_function import RastriginFunction

# Import helper functions
from menu.helper import get_algorithm_params, clear_screen

def continuous_menu():
    """Menu for continuous optimization problems."""
    while True:
        print("\n" + "="*80)
        print(" CONTINUOUS PROBLEMS MENU")
        print("="*80)
        print("\n  1. Full Analysis - All steps (Performance + Visualizations)")
        print("  2. Performance Computation - Compute metrics only")
        print("  3. Convergence Visualization")
        print("  4. Performance Analysis - Overview (All algorithms)")
        print("  5. Performance Analysis - Swarm vs Classical")
        print("  6. Performance Analysis - Swarm Only (Detailed)")
        print("  7. Sensitivity Analysis - Swarm algorithms")
        print("  8. Landscape Visualization")
        print("  0. Back to Main Menu")
        print("-" * 80)
        
        choice = input("\nChoice: ").strip()
        
        if choice == '0':
            break
        elif choice == '1':
            run_continuous_full_mode()
        elif choice == '2':
            run_performance_computation()
        elif choice == '3':
            run_convergence_visualization()
        elif choice == '4':
            run_comparative_analysis()
        elif choice == '5':
            run_paradigm_comparison()
        elif choice == '6':
            run_swarm_deep_dive()
        elif choice == '7':
            run_sensitivity_analysis()
        elif choice == '8':
            landscape_visualization()
        else:
            print("\n Invalid choice! Please try again.")
        
        if choice != '0':
            input("\nPress Enter to continue...")

def landscape_visualization():
    """Visualize 3D landscapes of optimization functions."""
    print("\n" + "="*80)
    print(" 3D Landscape Visualization")
    print("="*80)
    print("\nSelect function to visualize:")
    print("  1. Sphere")
    print("  2. Ackley")
    print("  3. Rastrigin")
    print("  4. All functions")
    print("  0. Back")
    
    function_choice_map = {
        '1': 'sphere',
        '2': 'ackley',
        '3': 'rastrigin',
    }
    
    choice = input("\nChoice (1-4): ").strip()
    
    if choice == '0':
        return
    elif choice in function_choice_map:
        function_name = function_choice_map[choice]
        print(f"\n Generating 3D landscape for {function_name.title()}...")
        plot_3d_surface(function_name)
        print(f" Visualization saved to: visualizations/continuous/landscapes/")
    elif choice == '4':
        print("\n Generating 3D landscapes for all functions...")
        generate_all_landscapes()
    else:
        print("\n Invalid choice! Please try again.")

def run_continuous_full_mode():
    """
    Full analysis mode for continuous problems.
    Run complete analysis with all 3 problems.
    """
    clear_screen()
    print("\n" + "="*80)
    print(" CONTINUOUS - Full Analysis Mode")
    print("="*80)
    print("\nThis mode will run:")
    print("  - Performance metrics computation (Sphere, Ackley, Rastrigin)")
    print("  - Convergence visualization")
    print("  - Overview performance analysis (All algorithms)")
    print("  - Swarm vs Classical comparison")
    print("  - Swarm-only performance analysis")
    print("\n  Estimated time: 10-15 minutes")
    print("-" * 80)
    
    confirm = input("\nProceed? (y/n) [y]: ").strip().lower() or 'y'
    if confirm != 'y':
        return
    
    problems = ['sphere', 'ackley', 'rastrigin']
    
    print("\n" + "="*80)
    print("STEP 1/5: Computing Performance Metrics")
    print("="*80)
    print("\n This may take 10-15 minutes...")
    
    # Get algorithm parameters (default: try YAML first, fallback to hardcoded)
    algorithms = get_algorithm_params()
    
    test_problems = [
        SphereFunction(dimensions=10),
        AckleyFunction(dimensions=10),
        RastriginFunction(dimensions=10)
    ]
    
    try:
        for problem in test_problems:
            benchmark_algorithms(algorithms, problem, n_runs=30, output_dir='results/continuous/performance')
        
        for problem_class in [SphereFunction, AckleyFunction, RastriginFunction]:
            benchmark_scalability(algorithms, problem_class, dimensions_list=[5, 10, 20, 30], n_runs=5, output_dir='results/continuous/performance')
        
        print("\n Performance metrics completed!")
    except Exception as e:
        print(f"\n Error: {e}")
    
    input("\nPress Enter to continue...")
    
    print("\n" + "="*80)
    print("STEP 2/5: Visualizing Convergence")
    print("="*80)
    try:
        visualize_all_convergence(problems=problems)
        print("\n Convergence visualization completed!")
    except Exception as e:
        print(f"\n Error: {e}")
    
    input("\nPress Enter to continue...")
    
    print("\n" + "="*80)
    print("STEP 3/5: Overview Performance Analysis")
    print("="*80)
    try:
        visualize_all_comparative(problems=problems)
        print("\n Overview performance analysis completed!")
    except Exception as e:
        print(f"\n Error: {e}")
    
    input("\nPress Enter to continue...")
    
    print("\n" + "="*80)
    print("STEP 4/5: Swarm vs Classical Comparison")
    print("="*80)
    try:
        visualize_all_paradigm_comparison(problems=problems)
        print("\n Swarm vs Classical comparison completed!")
    except Exception as e:
        print(f"\n Error: {e}")
    
    input("\nPress Enter to continue...")
    
    print("\n" + "="*80)
    print("STEP 5/5: Swarm Algorithms Deep Dive")
    print("="*80)
    try:
        visualize_all_swarm_metrics(problems=problems)
        print("\n Swarm-only performance analysis completed!")
    except Exception as e:
        print(f"\n Error: {e}")
    
    print("\n" + "="*80)
    print(" FULL ANALYSIS COMPLETE!")
    print("="*80)
    print("\n All results saved successfully!")
    print("\n Next steps:")
    print("   1. Review visualizations in visualizations/ directory")
    print("   2. Check metrics in results/ directory")
    print("   3. Use generated figures in your report")


def custom_mode():
    """Custom mode - Choose specific modules to run."""
    clear_screen()
    print("\n" + "="*80)
    print(" CUSTOM MODE - Select Modules")
    print("="*80)
    
    print("\nAvailable modules:")
    print("  1. Sensitivity Analysis")
    print("  2. Performance Metrics Computation")
    print("  3. Convergence Visualization")
    print("  4. Overview Performance (All Algorithms)")
    print("  5. Swarm vs Classical Comparison")
    print("  6. Swarm-Only Performance Analysis")
    print("  0. Back to main menu")
    
    choice = input("\nSelect module (0-6): ").strip()
    
    if choice == '0':
        return
    elif choice == '1':
        run_sensitivity_analysis()
    elif choice == '2':
        run_performance_computation()
    elif choice == '3':
        run_convergence_visualization()
    elif choice == '4':
        run_comparative_analysis()
    elif choice == '5':
        run_paradigm_comparison()
    elif choice == '6':
        run_swarm_deep_dive()
    elif choice == '7':
        landscape_visualization()
    else:
        print("\n Invalid choice!")
    
    input("\nPress Enter to return to main menu...")

def run_sensitivity_analysis():
    """Run parameter sensitivity analysis for swarm algorithms."""
    print("\n" + "="*80)
    print(" Parameter Sensitivity Analysis (Swarm Algorithms Only)")
    print("="*80)
    
    print("\nSelect swarm algorithm:")
    print("  1. FA (Firefly Algorithm)")
    print("  2. PSO (Particle Swarm Optimization)")
    print("  3. CS (Cuckoo Search)")
    print("  4. ABC (Artificial Bee Colony)")
    print("  5. All swarm algorithms")
    
    alg_choice = input("\nChoice (1-5): ").strip()
    
    alg_map = {
        '1': 'fa', '2': 'pso', '3': 'cs',
        '4': 'abc', '5': 'all'
    }
    
    if alg_choice not in alg_map:
        print("\n Invalid choice!")
        return
    
    try:
        if alg_choice == '5':
            print("\n Running sensitivity analysis for ALL swarm algorithms...")
            # Run only swarm algorithms: FA, PSO, CS, ABC
            for alg in ['fa', 'pso', 'cs', 'abc']:
                print(f"\n  Analyzing {alg.upper()}...")
                visualize_algorithm_results(
                    algorithm_name=alg,
                    input_dir='results/continuous/parameter_sensitivity',
                    output_dir='visualizations/continuous/parameter_sensitivity',
                    auto_run=True
                )
        else:
            alg_name = alg_map[alg_choice]
            print(f"\n Running sensitivity analysis for {alg_name.upper()}...")
            visualize_algorithm_results(
                algorithm_name=alg_name,
                input_dir='results/continuous/parameter_sensitivity',
                output_dir='visualizations/continuous/parameter_sensitivity',
                auto_run=True
            )
        
        print("\n Sensitivity analysis completed!")
        print(f" Results: results/continuous/parameter_sensitivity/")
        print(f" Visualizations: visualizations/continuous/parameter_sensitivity/")
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()


def run_performance_computation():
    """Run performance metrics computation."""
    print("\n" + "="*80)
    print(" Performance Metrics Computation")
    print("="*80)
    print("\n  This will take 10-15 minutes...")
    
    confirm = input("\nProceed? (y/n) [y]: ").strip().lower() or 'y'
    if confirm != 'y':
        return
    
    print("\n" + "="*80)
    print("COMPUTING PERFORMANCE METRICS")
    print("="*80)
    
    # Get algorithm parameters (default: try YAML first, fallback to hardcoded)
    algorithms = get_algorithm_params()
    
    # Define test problems
    problems = [
        SphereFunction(dimensions=10),
        AckleyFunction(dimensions=10),
        RastriginFunction(dimensions=10)
    ]
    
    try:
        # Benchmark on each problem
        for problem in problems:
            df, detailed = benchmark_algorithms(
                algorithms,
                problem,
                n_runs=30,
                output_dir='results/continuous/performance'
            )
            
            print(f"\n{problem.__class__.__name__} Results:")
            # Format numeric columns for better readability
            df_display = df.copy()
            for col in df_display.columns:
                if 'Fitness' in col:
                    df_display[col] = df_display[col].apply(lambda x: f'{x:.10f}' if pd.notna(x) else 'N/A')
                elif 'Time' in col:
                    df_display[col] = df_display[col].apply(lambda x: f'{x:.3f}' if pd.notna(x) else 'N/A')
                elif 'Convergence Speed' in col:
                    df_display[col] = df_display[col].apply(lambda x: f'{x:.1f}' if pd.notna(x) else 'N/A')
                elif 'Robustness' in col:
                    df_display[col] = df_display[col].apply(lambda x: f'{x:.2f}%' if pd.notna(x) else 'N/A')
            print(df_display.to_string(index=False))
        
        # Scalability analysis
        print("\n" + "="*70)
        for problem_class in [SphereFunction, AckleyFunction, RastriginFunction]:
            df_scalab, data_scalab = benchmark_scalability(
                algorithms,
                problem_class,
                dimensions_list=[5, 10, 20, 30],
                n_runs=5,
                output_dir='results/continuous/performance'
            )
            
            print(f"\n{problem_class.__name__} Scalability:")
            # Format numeric columns for better readability
            df_display = df_scalab.copy()
            for col in df_display.columns:
                if col == 'Dimension':
                    continue  # Keep dimension as integer
                elif 'Mean Fitness' in col or 'Fitness' in col:
                    df_display[col] = df_display[col].apply(lambda x: f'{x:.10f}' if pd.notna(x) else 'N/A')
                elif 'Mean Time' in col or 'Time' in col:
                    df_display[col] = df_display[col].apply(lambda x: f'{x:.3f}' if pd.notna(x) else 'N/A')
            print(df_display.to_string(index=False))
        
        print("\n" + "="*80)
        print("BENCHMARKING COMPLETE!")
        print("="*80)
        print("\n Results saved to: results/continuous/performance/")
        
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()


def run_convergence_visualization():
    """Run convergence visualization."""
    print("\n" + "="*80)
    print(" Convergence Visualization")
    print("="*80)
    
    problems = get_problem_selection()
    
    try:
        print("\n Creating convergence visualizations...")
        visualize_all_convergence(problems=problems, output_dir='visualizations/continuous/convergence')
        print("\n Convergence visualization completed!")
        print(" Visualizations: visualizations/continuous/convergence/")
    except Exception as e:
        print(f"\n Error: {e}")


def run_comparative_analysis():
    """Run overview performance analysis."""
    print("\n" + "="*80)
    print(" Overview Performance Analysis (All Algorithms)")
    print("="*80)
    
    problems = get_problem_selection()
    
    try:
        print("\n Creating overview performance visualizations...")
        visualize_all_comparative(problems=problems, results_dir='results/continuous/performance', output_dir='visualizations/continuous/performance/overview')
        print("\n Overview performance analysis completed!")
        print(" Visualizations: visualizations/continuous/performance/overview/")
    except Exception as e:
        print(f"\n Error: {e}")


def run_paradigm_comparison():
    """Run Swarm vs Classical comparison."""
    print("\n" + "="*80)
    print(" Swarm vs Classical Comparison")
    print("="*80)
    
    problems = get_problem_selection()
    
    try:
        print("\n Creating Swarm vs Classical comparison visualizations...")
        visualize_all_paradigm_comparison(problems=problems, results_dir='results/continuous/performance', output_dir='visualizations/continuous/performance/swarm_classic_comparison')
        print("\n Swarm vs Classical comparison completed!")
        print(" Visualizations: visualizations/continuous/performance/swarm_classic_comparison/")
    except Exception as e:
        print(f"\n Error: {e}")


def run_swarm_deep_dive():
    """Run swarm-only performance analysis."""
    print("\n" + "="*80)
    print(" Swarm-Only Performance Analysis")
    print("="*80)
    
    problems = get_problem_selection()
    
    try:
        print("\n Creating swarm-only performance visualizations...")
        visualize_all_swarm_metrics(problems=problems, results_dir='results/continuous/performance', output_dir='visualizations/continuous/performance/swarm_only')
        print("\n Swarm-only performance analysis completed!")
        print(" Visualizations: visualizations/continuous/performance/swarm_only/")
    except Exception as e:
        print(f"\n Error: {e}")


def get_problem_selection():
    """Get problem selection from user."""
    print("\nSelect problems:")
    print("  1. Sphere only")
    print("  2. Ackley only")
    print("  3. Rastrigin only")
    print("  4. All problems (Sphere + Ackley + Rastrigin)")
    
    choice = input("\nChoice (1-4) [4]: ").strip() or '4'
    
    if choice == '1':
        return ['sphere']
    elif choice == '2':
        return ['ackley']
    elif choice == '3':
        return ['rastrigin']
    elif choice == '4':
        return ['sphere', 'ackley', 'rastrigin']
    else:
        print("\n Invalid choice!")
        return []


def view_results():
    """Show results directories for continuous and discrete."""
    clear_screen()
    print("\n" + "="*80)
    print(" View Results")
    print("="*80)
    
    print("\n CONTINUOUS RESULTS:")
    print("  1. results/continuous/performance/")
    print("  2. visualizations/continuous/convergence/")
    print("  3. visualizations/continuous/performance/overview/")
    print("  4. visualizations/continuous/performance/swarm_classic_comparison/")
    print("  5. visualizations/continuous/performance/swarm_only/")
    print("  6. visualizations/continuous/parameter_sensitivity/")
    print("  7. visualizations/continuous/landscapes/")
    
    print("\n DISCRETE RESULTS:")
    print("  8. results/discrete/convergence/")
    print("  9. results/discrete/performance/")
    print(" 10. visualizations/discrete/convergence/")
    print(" 11. visualizations/discrete/parameter_sensitivity/")
    
    print("\n" + "-" * 80)
    
    # Check which directories exist
    continuous_dirs = [
        'results/continuous/performance',
        'visualizations/continuous/convergence',
        'visualizations/continuous/performance/overview',
        'visualizations/continuous/performance/swarm_classic_comparison',
        'visualizations/continuous/performance/swarm_only',
        'visualizations/continuous/parameter_sensitivity',
        'visualizations/continuous/landscapes'
    ]
    
    discrete_dirs = [
        'results/discrete/convergence',
        'results/discrete/performance',
        'visualizations/discrete/convergence',
        'visualizations/discrete/parameter_sensitivity'
    ]
    
    print("\n CONTINUOUS - Directory status:")
    for dir_path in continuous_dirs:
        path = Path(dir_path)
        status = "EXISTS" if path.exists() else "NOT FOUND"
        print(f"  {dir_path}: {status}")
    
    print("\n DISCRETE - Directory status:")
    for dir_path in discrete_dirs:
        path = Path(dir_path)
        status = "EXISTS" if path.exists() else "NOT FOUND"
        print(f"  {dir_path}: {status}")
    
    print("\n" + "-" * 80)
    print("\n To open a directory, navigate manually to the paths above.")
