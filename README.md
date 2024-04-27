# Sudoku-Solver

Sudoku Solver README

Introduction:
This program is a Sudoku solver that can be run in the terminal. It has several features, including recursive solving, explaining the steps taken to solve a puzzle, saving solutions to a file, and giving hints to the user.

Prerequisites:
Python 3.x
NumPy

Installation:
Download or clone the repository.
Install the required packages using pip: pip install numpy

Usage:
1. Open a terminal and navigate to the directory where the repository is located.
2. To run the recursive solver on a grid, type: python main.py grid_input --recursive, where grid_input is the name of the file containing the Sudoku grid you want to solve.
3. To explain the steps taken to solve a grid, type: python main.py grid_input --explain, where grid_input is the file with the grid you want to solve.
4. To save the solution to a file, type: python main.py grid_input --file grid_input grid_output, where grid_output is the name you want to give to the file that will store the solution.
5. To get hints for a grid, type: python main.py grid_input --hint N, where N is the number of hints you want to get.
6. This program includes a profiling feature that measures the performance of the solver(s) for grids of different sizes and difficulties. To run the profiler, type: python main.py --profile.
7. To run the wavefront solver on a grid, type: python main.py grid_input --wavefront

Special cases:
When using grids that aren't 3x3 we need to do something different, 
1. Input the following: python main.py grid_input --subgrid_cols N --subgrid_rows N, for example for a 3x2 sudoku you should put --subgrid_cols 3 --subgrid_rows 2.
2. Then if you want to run the flags shown above just put the flag after it e.g python main.py grid_input --subgrid_cols N --subgrid_rows N --explain.

Note: The --explain and --file flags can be used together and the --explain and --hint N flags can be used together, but all 3 of them can't be used together. The profile flag runs on its own and measures the grids that are already in the code, so feel free to change them with you own grids to test. The other flags solve the grid one by one in the terminal as in you would have to put one grid_input run that and then after put the next one and run the whole thing again.

Conclusion
This program is a useful tool for solving Sudoku puzzles and can be easily used in the terminal with a variety of features.
