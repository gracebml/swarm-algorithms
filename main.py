"""
Comparing Swarm-based (PSO, FA, CS, ABC) vs Classical (GA, SA) algorithms 
- Continuous problems: Sphere, Ackley, Rastrigin
- Discrete problems: TSP (ACO, A*)
"""

import sys
from pathlib import Path

# Add src to path
src_dir = Path(__file__).parent / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# Add menu to path
menu_dir = Path(__file__).parent / 'menu'
if str(menu_dir) not in sys.path:
    sys.path.insert(0, str(menu_dir))

# Import menu modules
from menu.helper import clear_screen, print_header, print_menu
from menu.continuous_mode import continuous_menu, view_results
from menu.discrete_mode import discrete_menu


def main():
    while True:
        clear_screen()
        print_header()
        print_menu()
        
        choice = input("\nEnter your choice (0-3): ").strip()
        
        if choice == '0':
            print("\n Thank you for using AI Fundamentals Lab 1!")
            print(" Goodbye!\n")
            break
        elif choice == '1':
            # Continuous problems menu
            continuous_menu()
        elif choice == '2':
            # Discrete problems menu
            discrete_menu()
        elif choice == '3':
            # View results
            view_results()
            input("\nPress Enter to return to main menu...")
        else:
            print("\n Invalid choice! Please enter 0-3.")
            input("\nPress Enter to continue...")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Program interrupted by user.")
        print(" Goodbye!\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
        sys.exit(1)
