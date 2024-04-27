#!/usr/bin/env python
import random
import time
import numpy as np
import argparse
import copy
import matplotlib.pyplot as plt
import statistics
import timeit


# grids for --profile flag

grid1 = [
		[1, 0, 4, 2],
		[4, 2, 1, 3],
		[2, 1, 3, 4],
		[3, 4, 2, 1]]

grid2 = [
		[1, 0, 4, 2],
		[4, 2, 1, 3],
		[2, 1, 0, 4],
		[3, 4, 2, 1]]

grid3 = [
		[1, 0, 4, 2],
		[4, 2, 1, 0],
		[2, 1, 0, 4],
		[0, 4, 2, 1]]

grid4 = [
		[1, 0, 4, 2],
		[0, 2, 1, 0],
		[2, 1, 0, 4],
		[0, 4, 2, 1]]

grid5 = [
		[1, 0, 0, 2],
		[0, 0, 1, 0],
		[0, 1, 0, 4],
		[0, 0, 0, 1]]

grid6 = [
		[0, 0, 6, 0, 0, 3],
		[5, 0, 0, 0, 0, 0],
		[0, 1, 3, 4, 0, 0],
		[0, 0, 0, 0, 0, 6],
		[0, 0, 1, 0, 0, 0],
		[0, 5, 0, 0, 6, 4]]

grid7 = [
        [0, 3, 0, 4, 0, 0],
        [0, 0, 5, 6, 0, 3],
        [0, 0, 0, 1, 0, 0],
        [0, 1, 0, 3, 0, 5],
        [0, 6, 4, 0, 3, 1],
        [0, 0, 1, 0, 4, 6]]

grid8 = [
        [9, 0, 6, 0, 0, 1, 0, 4, 0],
        [7, 0, 1, 2, 9, 0, 0, 6, 0],
        [4, 0, 2, 8, 0, 6, 3, 0, 0],
        [0, 0, 0, 0, 2, 0, 9, 8, 0],
        [6, 0, 0, 0, 0, 0, 0, 0, 2],
        [0, 9, 4, 0, 8, 0, 0, 0, 0],
        [0, 0, 3, 7, 0, 8, 4, 0, 9],
        [0, 4, 0, 0, 1, 3, 7, 0, 6],
        [0, 6, 0, 9, 0, 0, 1, 0, 8]]

grid9 = [
        [0, 0, 0, 2, 6, 0, 7, 0, 1],
        [6, 8, 0, 0, 7, 0, 0, 9, 0],
        [1, 9, 0, 0, 0, 4, 5, 0, 0],
        [8, 2, 0, 1, 0, 0, 0, 4, 0],
        [0, 0, 4, 6, 0, 2, 9, 0, 0],
        [0, 5, 0, 0, 0, 3, 0, 2, 8],
        [0, 0, 9, 3, 0, 0, 0, 7, 4],
        [0, 4, 0, 0, 5, 0, 0, 3, 6],
        [7, 0, 3, 0, 1, 8, 0, 0, 0]]

grid10 =[
        [0, 2, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 6, 0, 4, 0, 0, 0, 0],
        [5, 8, 0, 0, 9, 0, 0, 0, 3],
        [0, 0, 0, 0, 0, 3, 0, 0, 4],
        [4, 1, 0, 0, 8, 0, 6, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 9, 5],
        [2, 0, 0, 0, 1, 0, 0, 8, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 3, 1, 0, 0, 8, 0, 5, 7]]

grid11 = [
        [0, 0, 0, 6, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 5, 0, 1],
        [3, 6, 9, 0, 8, 0, 4, 0, 0],
        [0, 0, 0, 0, 0, 6, 8, 0, 0],
        [0, 0, 0, 1, 3, 0, 0, 0, 9],
        [4, 0, 5, 0, 0, 9, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 3, 0, 0],
        [0, 0, 6, 0, 0, 7, 0, 0, 0],
        [1, 0, 0, 3, 4, 0, 0, 0, 0]]

grids = [(grid1, 2, 2), (grid2, 2, 2), (grid3, 2, 2), (grid4, 2, 2), (grid5, 2, 2), (grid6, 2, 3), (grid7, 2, 3), (grid8, 3, 3), (grid9, 3, 3), (grid10, 3, 3), (grid11, 3, 3)]

'''
===================================
DO NOT CHANGE CODE BELOW THIS LINE
===================================
'''

def parse_arguments():
    # create an ArgumentParser object with the given description
    parser = argparse.ArgumentParser(description='Sudoku Solver')

    # add an argument to the parser to specify the input file containing the puzzle
    parser.add_argument('input_file', nargs='?', default=None, help='input file containing the sudoku puzzle')

    # add an optional argument to the parser to use the wavefront function
    parser.add_argument('--wavefront', action="store_true", help="use wavefront function")

    # add an optional argument to the parser to use the recursive_solve function
    parser.add_argument("--recursive", action="store_true", help="use recursive_solve function")

    # add an optional argument to the parser to show detailed steps to solve the puzzle
    parser.add_argument('--explain', action='store_true', help='show detailed steps to solve the puzzle')

    # add an optional argument to the parser to specify the number of rows in a subgrid
    parser.add_argument('--subgrid_rows', type=int, default=3, help='number of rows in a subgrid')

    # add an optional argument to the parser to specify the number of columns in a subgrid
    parser.add_argument('--subgrid_cols', type=int, default=3, help='number of columns in a subgrid')

    # add an optional argument to the parser to specify the input and output file paths
    parser.add_argument('--file', nargs=2, metavar=('INPUT', 'OUTPUT'), help='input and output file paths')

    # add an optional argument to the parser to return a partially solved puzzle with N values filled in
    parser.add_argument('--hint', type=int, help='return a partially solved puzzle with N values filled in')

    # add an optional argument to the parser to return a graph comparing different solvers
    parser.add_argument('--profile', action="store_true", help='return graph comparing solver performances')
    return parser.parse_args()


def read_puzzle(file_path):
    # Open the file located at file_path in read-only mode
    with open(file_path, 'r') as f:
        # Read all lines from the file and store them in the 'lines' variable
        lines = f.readlines()

    # Create an empty list to hold the puzzle
    puzzle = []
    # Iterate over each line in 'lines'
    for line in lines:
        # Create an empty list to hold the integers in the line
        row = []
        # Split the line on commas, strip whitespace from each element, convert to int, and append to 'row'
        for c in line.strip().split(','):
            row.append(int(c))
        # Append the row to the puzzle
        puzzle.append(row)

    # Return the completed puzzle
    return puzzle


def is_valid(puzzle, row, col, num, subgrid_rows, subgrid_cols):
    # Check row
    if num in puzzle[row]:
        return False

    # Check column
    for i in range(subgrid_rows * subgrid_cols):
        if puzzle[i][col] == num:
            return False

    # Check square
    row_start = (row // subgrid_rows) * subgrid_rows
    col_start = (col // subgrid_cols) * subgrid_cols
    for i in range(row_start, row_start + subgrid_rows):
        for j in range(col_start, col_start + subgrid_cols):
            if puzzle[i][j] == num:
                return False

    return True


def is_complete(puzzle):
    # Loop through each element of the puzzle
    for i in range(len(puzzle)):
        for j in range(len(puzzle[0])):
            # If an element is equal to 0, the puzzle is not complete
            if puzzle[i][j] == 0:
                return False
    # If no element is equal to 0, the puzzle is complete
    return True


def find_empty_cell(puzzle, subgrid_rows, subgrid_cols):
    # iterate over each row in the puzzle
    for i in range(len(puzzle)):
        # iterate over each column in the puzzle
        for j in range(len(puzzle[0])):
            # if a cell is empty (i.e., contains a 0), return its coordinates
            if puzzle[i][j] == 0:
                return i, j
    # if there are no empty cells in the puzzle, return None
    return None


def solve_puzzle(puzzle, subgrid_rows, subgrid_cols, explain=False, hint=None, hint_count=0):
    # Base case: the puzzle is complete
    if is_complete(puzzle):
        return puzzle
    # Find the next empty cell in the puzzle
    cell = find_empty_cell(puzzle, subgrid_rows, subgrid_cols)
    row, col = cell

    hint_locations = None
    # If hints are requested, keep track of the locations where they are given
    if hint:
        if hint_count >= hint:
            return puzzle[:]

        hint_locations = []
    # Try each possible number in the cell and recursively solve the resulting puzzle
    for num in range(1, len(puzzle) + 1):
        if is_valid(puzzle, row, col, num, subgrid_rows, subgrid_cols):
            puzzle[row][col] = num
            # If explanation is requested, print out the updated puzzle after each move
            if explain:
                print(f"Put {num} in location ({row+1}, {col+1})")
                for r in range(len(puzzle)):
                    row_str = ''
                    for c in range(len(puzzle[0])):
                        if r == row and c == col:
                            row_str += f"[{puzzle[r][c]}]"
                        else:
                            row_str += f" {puzzle[r][c]} "
                    print(row_str)
                print('\n')
            # If hints are requested, keep track of the number of hints shown so far
            if hint and hint_count < hint:
                hint_count += 1
                hint_locations.append((row, col))
            # Recursively solve the resulting puzzle
            result = solve_puzzle(puzzle, subgrid_rows, subgrid_cols, explain, hint, hint_count)
            if result:
                return result
            # If the recursion does not lead to a solution, backtrack and try the next possible number
            puzzle[row][col] = 0
    # If all possible numbers have been tried and none lead to a solution, return None
    # If hints have been requested and the maximum number of hints has been reached, print out the hint locations
    if hint_locations and hint_count == hint:
        print(f"\nHints: {hint}")
        for location in hint_locations:
            row, col = location
            print(f" - Put {puzzle[row][col]} in location ({row+1}, {col+1})")

    return None


def print_puzzle(puzzle):
    # Iterate through the rows of the puzzle
    for i in range(len(puzzle)):
        row_str = ''
        # Iterate through the columns of the puzzle
        for j in range(len(puzzle[0])):
            # Add the value of the cell to the row string, followed by a space
            row_str += f"{puzzle[i][j]} "
            # If the current column is a multiple of 3 and not the last column, add a vertical bar to separate subgrids
            if (j+1) % 3 == 0 and j < 8:
                row_str += '| '
        # Print the row string
        print(row_str)
        # If the current row is a multiple of 3 and not the last row, print a horizontal line to separate subgrids
        if (i+1) % 3 == 0 and i < 8:
            print('-'*22)


def check_section(section, n):
    # Check if section contains only unique values (no duplicates)
    # and if the sum of the section equals the sum of values from 0 to n
    if len(set(section)) == len(section) and sum(section) == sum([i for i in range(n + 1)]):
        return True
    return False


def get_squares(grid, n_rows, n_cols):
    # Create an empty list to store all the squares
    squares = []
    
    # Loop through the number of columns
    for i in range(n_cols):
        # Calculate the start and end indices of the rows for the current square
        rows = (i * n_rows, (i + 1) * n_rows)
        
        # Loop through the number of rows
        for j in range(n_rows):
            # Calculate the start and end indices of the columns for the current square
            cols = (j * n_cols, (j + 1) * n_cols)
            
            # Create an empty list to store the current square
            square = []
            
            # Loop through the rows of the current square
            for k in range(rows[0], rows[1]):
                # Extract the cells of the current square from the grid
                line = grid[k][cols[0]:cols[1]]
                square += line
            
            # Add the current square to the list of squares
            squares.append(square)

    # Return the list of squares
    return squares


def check_solution(grid, n_rows, n_cols):
    '''
 	This function is used to check whether a sudoku board has been correctly solved

 	args: grid - representation of a suduko board as a nested list.
 	returns: True (correct solution) or False (incorrect solution)
 	'''
    n = n_rows * n_cols

    for row in grid:
        if check_section(row, n) == False:
            return False

    for i in range(n_rows ** 2):
        column = []
        for row in grid:
            column.append(row[i])

        if check_section(column, n) == False:
            return False

    squares = get_squares(grid, n_rows, n_cols)
    for square in squares:
        if check_section(square, n) == False:
            return False

    return True


def find_empty(grid):
    '''
    This function returns the index (i, j) to the first zero element in a sudoku grid
    If no such element is found, it returns None
    args: grid
    return: A tuple (i,j) where i and j are both integers, or None
    '''

    for i in range(len(grid)):
        row = grid[i]
        for j in range(len(row)):
            if grid[i][j] == 0:
                return (i, j)

    return None


def find_empty_all(grid):
    '''
    This function finds all the locations on the sudoku grid which is empty,
    represented by a 0 by looping through all the positions.
    args:  grid - sudoku grid in the form of a nested list
    return: List of coordinates in the order they were found
 	'''
    # creates a list which will hold all of the coordinates representing empty positions
    empty = []
    # loops through all of the rows and columns using two variables i and j where
    # each loop goes up to the largest value in its respective direction
    for i in range(len(grid)):
        row = grid[i]
        for j in range(len(row)):
    # checks if each set of i and j coordinates is equal to 0, and if they are
    # they are saved and added to the list created at the start.
            if grid[i][j] == 0:
                empty.append((i, j))
    # the list of coordinates is returned
    return empty

def remove_numbers(grid_range, grid, n_rows, n_cols):
    '''
    This function removes numbers from the sudoku possible number range
    args:   grid_range - a given row/column/square of sudoku, from which we remove numbers in a list of possibilities for 0s
            grid - sudoku grid in the form of a nested list
            n_rows - the number of rows of a sudoku grid
            n_cols - the number of columns of the sudoku grid
    return: the updated list containing the range of possible numbers for a given 0 space in the grid
    '''
    n = n_rows * n_cols  # max number range of a given sudoku
    num_range = list(range(1, n + 1))  # creating list for number range

    for element in grid_range:
        if element in num_range:
            num_range.remove(element)
    return num_range

def coordinate_value_finder(n,grid,row,col,n_rows,n_cols,transpose):
    '''
    This function creates a list of values which could go in each position, by checking
    the rows, columns and squares it removes the values which are impossible leaving
    far fewer options.
    args:   grid - sudoku grid in the form of a nested list
            n - values which can be inputted into each space
            row - represents the location on the horizontal
            col - represents the location on the vertical
            n_rows - the number of rows of the sudoku grid
            n_cols - the number of columns of the sudoku grid
            transpose - a list containing each element in a column
    return: Returns possible values which can be inputted ine ach location
    '''
    # sudoku number set in any given row/column/square
    num_range = list(range(1, n + 1))

    # iterates through row that has an empty space to remove those numbers from set of possibilities
    for element in grid[row]:
        if element in num_range:
            num_range.remove(element)

    # iterates through column that has empty space via transpose to remove those numbers from set of possibilities
    for element1 in transpose[col]:
        if element1 in num_range:
            num_range.remove(element1)

    # eliminating numbers in same square as a 0
    square_row = (row // n_rows) * n_rows  # coordinate of topmost row of square
    square_col = (col // n_cols) * n_cols  # coordinate of leftmost column of square
    square_coords = [(r, c) for r in range(square_row, square_row + n_rows)
                      # makes coordinate list of all numbers in same square as 0
                      for c in range(square_col, square_col + n_cols)]

    # extracting the corresponding numbers from the respective square
    square_list = [grid[r][c] for r, c in square_coords]

    # iterates through numbers in same square as 0, removing them from set of possibilities
    for num in square_list:
        if num in num_range:
            num_range.remove(num)
    return num_range

def least_possibilities(grid, n_rows, n_cols):
    '''
    This function looks through all the empty locations and finds the coordinates
    with the least possibilities, giving the recursive solve the best place to 
    begin.
    args:   grid - sudoku grid in the form of a nested list
            n_rows - the number of rows of the sudoku grid
            n_cols - the number of columns of the sudoku grid
    return: The coordinates which have the fewest possibilities.
    '''

    # Finds the range of values which can actually be inputted
    n = n_rows * n_cols
    # Calls the find_empty_all function which gives a list of all the coordinates
    # which have a 0 in (empty).
    empty = find_empty_all(grid)
    # checks if there are any empty positions left in the sudoku grid
    # finds transpose of a sudoku grid 
    transpose = np.transpose(grid)
    if not empty:
    # returns 0 if the grid has been solved
        return 0
    # minimum number of possible values for the empty cells
    least_options = 0
    # checks if the current position is empty or not and continues unitll it finds a non empty
    for current_pos in empty:
        while current_pos == 0 :
            print()
    # When the next coordinate is found row and col are assigned to both the x and y coordinate
        row, col = current_pos
        
    # Checks if there are any coordinates in empty, if there are not it returns its current position
        if empty == []:
    # If there aren't any then find_pos_values is called where it finds the possible values
            coordinate_value_finder(n,grid,row,col,n_rows,n_cols,transpose)
            return current_pos
    # If least options is empty or has a value less than previously determined it gets re-defined to the smaller value
    # and the new best position is found.
        if least_options == 0 or len(coordinate_value_finder(n,grid,row,col,n_rows,n_cols,transpose)) < least_options:
            least_options = len(coordinate_value_finder(n,grid,row,col,n_rows,n_cols,transpose))
            #print(least_options)
            best_pos = current_pos
    # returns the position with the fewest options to be inputted after all corrdinates have been looked at
    return best_pos


def recursive_solve(grid, n_rows, n_cols, explain = False):
    '''
    This function uses recursion to exhaustively search all possible solutions to a grid
    until the solution is found
    args:   grid - sudoku grid in the form of a nested list
            n_rows - the number of rows of the sudoku grid
            n_cols - the number of columns of the sudoku grid
            last_pos - the last position filled in the grid
    return: A solved grid (as a nested list), or None
    '''
    # n is the maximum integer considered in this board
    n = n_rows * n_cols

    # Find an empty place in the grid (returns position)
    empty = least_possibilities(grid, n_rows, n_cols)
    
    # finds transpose of a sudoku grid 
    transpose = np.transpose(grid)

    # If there's no empty places left, check if we've found a solution
    if not empty:
        # If the solution is correct, return it.
        if check_solution(grid, n_rows, n_cols):
            return grid
        else:
            # If the solution is incorrect, return None
            return None
    else:
        row, col = empty

    # iterates through numbers in same square as 0, removing them from set of possibilities
    for num in coordinate_value_finder(n,grid,row,col,n_rows,n_cols,transpose):
        if num in coordinate_value_finder(n,grid,row,col,n_rows,n_cols,transpose):
            coordinate_value_finder(n,grid,row,col,n_rows,n_cols,transpose).remove(num)

    # Loop through possible values
    for i in coordinate_value_finder(n,grid,row,col,n_rows,n_cols,transpose):
        # Place the value into the grid
        grid[row][col] = i
        # Recursively solve the grid
        ans = recursive_solve(grid, n_rows, n_cols)
        # If we've found a solution, return it
        if ans:
            return ans

    # If we get here, we've tried all possible values. Return none to indicate the previous value is incorrect. 
    grid[row][col] = 0
    return None

def fill_board_randomly(grid, n_rows, n_cols):
    '''
    This function will fill an unsolved Sudoku grid with random numbers
    args: grid, n_rows, n_cols
    return: A grid with all empty values filled in
    '''
    n = n_rows * n_cols
    # Make a copy of the original grid
    filled_grid = copy.deepcopy(grid)

    # Loop through the rows
    for i in range(len(grid)):
        # Loop through the columns
        for j in range(len(grid[0])):
            # If we find a zero, fill it in with a random integer
            if grid[i][j] == 0:
                filled_grid[i][j] = random.randint(1, n)

    return filled_grid

def random_solve(grid, n_rows, n_cols, max_tries=50000):
    '''
    This function uses random trial and error to solve a Sudoku grid
    args: grid, n_rows, n_cols, max_tries
    return: A solved grid (as a nested list), or the original grid if no solution is found
    '''
    for i in range(max_tries):
        possible_solution = fill_board_randomly(grid, n_rows, n_cols)
        if check_solution(possible_solution, n_rows, n_cols):
            return possible_solution

    return grid

def get_possible_values(grid, n_rows, n_cols):
    '''
    This function returns a list representing the possible values for each empty grid location
    '''

    n = n_rows * n_cols
    possible_values = [[[] for numbers in range(n + 1)] for numbers in range(n + 1)]

    for row in range(n):
        for col in range(n):
            if grid[row][col] == 0:
                row_values = set(grid[row])
                col_values = set(grid[row_index][col] for row_index in range(n))
                square_values = set(grid[(row//n_rows)*n_rows + row_index][(col//n_cols)*n_cols + col_index]
                                    for row_index in range(n_rows) for col_index in range(n_cols))

                possible_values[row][col] = list(set(range(1, n+1)) - row_values - col_values - square_values)

    return possible_values

def check_values(grid, value, row, col, n_rows, n_cols):
    '''
    This function checks through each row, column and square of a grid to ensure there are no repeated values
    args:   value - the value being checked within the grid
            row - the row being checked through within the grid
            col - the column being checked through within the grid
    '''
    n = n_rows * n_cols
    # check for duplicates in row
    if value in grid[row]:
        return True

    # check for duplicates in column
    for row_index in range(n):
        if grid[row_index][col] == value:
            return True

    # check for duplicates in square
    square_row = (row // n_rows) * n_rows
    square_col = (col // n_cols) * n_cols
    for row_index in range(square_row, square_row + n_rows):
        for col_index in range(square_col, square_col + n_cols):
            if grid[row_index][col_index] == value:
                return True
    return False


def wavefront_solve(grid, n_rows, n_cols, explain = False):
    '''
    This function solves a Sudoku puzzle using the Wavefront propagation algorithm.

    args: grid - representation of a Sudoku board as a nested list.
          n_rows - number of rows in each square of the Sudoku board.
          n_cols - number of columns in each square of the Sudoku board.

    returns: the solved Sudoku board as a nested list.
    '''

    n = n_rows * n_cols

    def sudoku_solve():
        min_possible_values = n + 1
        min_row = -1
        min_col = -1

        # find the empty location with the smallest number of possible values
        for row in range(n):
            for col in range(n):
                if grid[row][col] == 0:
                    num_possible_values = len(set(get_possible_values(grid, n_rows, n_cols)[row][col]))
                    if num_possible_values == 0:
                        # if there's no possible values for this location go back to the previous and try a different value
                        return False
                    elif num_possible_values <= min_possible_values:
                        min_possible_values = num_possible_values
                        min_row, min_col = row, col

        if check_solution(grid, n_rows, n_cols):
            # all locations are filled and the sudoku is solved
            return True

        possible_values = get_possible_values(grid, n_rows, n_cols)[min_row][min_col]
        grid[min_row][min_col] = 0
        for value in possible_values:
            if not check_values(grid, value, min_row, min_col, n_rows, n_cols):
                # input the values into the grid
                grid[min_row][min_col] = value

                # call the solve function to repeat steps and continue filling the grid
                if sudoku_solve():
                    return True

                # if a value is incorrect undo and retry
                grid[min_row][min_col] = 0

        return False

    sudoku_solve()
    return grid

# grid lists to be used in grid_categoriser function

grids_2x2 = []
grids_3x2 = []
grids_3x3 = []

def grid_categoriser(grid_list):
    '''
    takes grid index, extracts said parameters and categorises each grid by size into one of three lists
    :param grid_list: list of sudoku grid indexes in form (grid, number_of_rows, number_of_columns)
    :return: three lists with each category of sudoku
    '''
    for grid in grid_list:
        if grid[1] and grid[2] == 2:
            grids_2x2.append(grid)
        elif grid[1] == 2 and grid[2] == 3:
            grids_3x2.append(grid)
        elif grid[1] and grid[2] == 3:
            grids_3x3.append(grid)
    return grids_2x2, grids_3x2, grids_3x3


def recursive_solve_times(grid_list):
    '''
    :param grid_list: list of sudoku grids
    :return: times taken for each grid to be solved using the recursive solver
    '''
    time_list = []
    original_grids = []
    for grid in grid_list:
        # Make a copy of the original grid
        filled_grid = copy.deepcopy(grid)
        original_grids.append(filled_grid)
        for grid in original_grids:
            elapsed_time = timeit.timeit(lambda: recursive_solve(grid[0], grid[1], grid[2]), number=1)
            time_list.append(elapsed_time)
    return time_list


def wavefront_solve_times(grid_list):
    '''
    :param grid_list: list of sudoku grids
    :return: times taken for each grid to be solved using the wavefront solver
    '''
    time_list = []
    original_grids = []
    for grid in grid_list:
        # Make a copy of the original grid
        filled_grid = copy.deepcopy(grid)
        original_grids.append(filled_grid)
        for grid in original_grids:
            elapsed_time = timeit.timeit(lambda: wavefront_solve(grid[0], grid[1], grid[2]), number=1)
            time_list.append(elapsed_time)
    return time_list


def rand_solve_times(grid_list):
    '''
    :param grid_list: list of sudoku grids
    :return: times taken for each grid to be solved using the random solver
    '''
    time_list = []
    original_grids = []
    for grid in grid_list:
        # Make a copy of the original grid
        filled_grid = copy.deepcopy(grid)
        original_grids.append(filled_grid)
        for grid in original_grids:
            elapsed_time = timeit.timeit(lambda: random_solve(grid[0], grid[1], grid[2]), number=1)
            time_list.append(elapsed_time)
    return time_list


def solver_performance(grids):
    '''
    :param grids: list of sudoku grids
    :return: graph comparing performance of the random, recursive and wavefront sudoku solvers on different sudoku grid sizes
    '''
    # running grid categorising function to then run solvers on grid categories by grid size
    grid_categoriser(grids)

    # recursive solver times for each grid size
    recsolve_times_2x2_grids = statistics.mean(recursive_solve_times(grids_2x2))
    recsolve_times_3x2_grids = statistics.mean(recursive_solve_times(grids_3x2))
    recsolve_times_3x3_grids = statistics.mean(recursive_solve_times(grids_3x3))

    # list of the average time taken for each grid size
    recsolve_times_all = [recsolve_times_2x2_grids, recsolve_times_3x2_grids, recsolve_times_3x3_grids]

    # wavefront solver times for each grid size
    wavesolve_times_2x2_grids = statistics.mean(wavefront_solve_times(grids_2x2))
    wavesolve_times_3x2_grids = statistics.mean(wavefront_solve_times(grids_3x2))
    wavesolve_times_3x3_grids = statistics.mean(wavefront_solve_times(grids_3x3))

    # list of average time taken for each grid size
    wavesolve_times_all = [wavesolve_times_2x2_grids, wavesolve_times_3x2_grids, wavesolve_times_3x3_grids]

    # random solver times for each grid size
    randsolve_times_2x2_grids = statistics.mean(rand_solve_times(grids_2x2))
    randsolve_times_3x2_grids = statistics.mean(rand_solve_times(grids_3x2))
    randsolve_times_3x3_grids = statistics.mean(rand_solve_times(grids_3x3))

    # list of average time taken for each grid size
    randsolve_times_all = [randsolve_times_2x2_grids, randsolve_times_3x2_grids, randsolve_times_3x3_grids]

    # bar chart for comparing times taken to solve different sized grids by different solvers
    bar_width = 0.25
    recplot = np.arange(len(recsolve_times_all))
    waveplot = [x + bar_width for x in recplot]
    randplot = [x + bar_width for x in waveplot]

    # creating bars for each solver based on grid size
    plt.bar(recplot, recsolve_times_all, color='b', width=bar_width, edgecolor='white', label='Recursive solver')
    plt.bar(waveplot, wavesolve_times_all, color='g', width=bar_width, edgecolor='white', label='Wavefront solver')
    plt.bar(randplot, randsolve_times_all, color='r', width=bar_width, edgecolor='white', label='Random solver (does not solve,\ntime taken for 50000 iterations)')
    plt.yscale('log')  # times taken range very wide, log scale allows all results to be visible

    plt.xlabel('Grid Size')
    plt.ylabel('Time (seconds)')
    plt.title('Sudoku Solver Performance')
    plt.xticks([r + bar_width for r in range(len(recsolve_times_all))], ['2x2', '3x2', '3x3'])
    plt.legend()

    plt.show()

def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Record the start time
    start_time = time.time()
    if args.input_file:
        # read the puzzle from the input file
        puzzle = read_puzzle(args.input_file)
        # Print the initial puzzle
        print("Initial puzzle:")
        print_puzzle(puzzle)
        # If hint is specified and explain is not, generate hints
        if args.hint and not args.explain:
            hint_puzzle = [row[:] for row in puzzle]  # create a copy of the puzzle
            hint_count = args.hint  # number of hints to generate
            hint_solution = solve_puzzle(hint_puzzle, args.subgrid_rows, args.subgrid_cols)  # solve the puzzle
            hints_found = 0
            for i in range(len(puzzle)):
                for j in range(len(puzzle[0])):
                    if hint_count == 0:
                        break
                    # If the cell is empty in the original puzzle and has a value in the solution, add a hint
                    if puzzle[i][j] == 0 and hint_solution[i][j] != 0:
                        if hints_found < args.hint:
                            print(f"Hint {hints_found + 1}: Put {hint_solution[i][j]}")
                        puzzle[i][j] = hint_solution[i][j]
                        hint_count -= 1
                        hints_found += 1
            # Record the end time
            end_time = time.time()
            # Print the puzzle with hints
            print(f"Puzzle with {args.hint} hints:")
            print_puzzle(puzzle)
            print(f"Time taken: {end_time - start_time:.5f} seconds")
        else:
            # If hint is specified and explain is also specified, generate hints and explanations
            if args.hint and args.explain:
                hint_puzzle = [row[:] for row in puzzle]  # create a copy of the puzzle
                hint_count = args.hint  # number of hints to generate
                hint_solution = solve_puzzle(hint_puzzle, args.subgrid_rows, args.subgrid_cols)  # solve the puzzle
                hints_found = 0
                hint_locations = []

                for i in range(len(puzzle)):
                    for j in range(len(puzzle[0])):
                        if hint_count == 0:
                            break
                        # If the cell is empty in the original puzzle and has a value in the solution, add a hint and explanation
                        if puzzle[i][j] == 0 and hint_solution[i][j] != 0:
                            if hints_found < args.hint:
                                print(f"Hint {hints_found + 1}: Put {hint_solution[i][j]} in location ({i+1}, {j+1})")
                                hint_locations.append((i, j))
                            puzzle[i][j] = hint_solution[i][j]
                            hint_count -= 1
                            hints_found += 1
                # Record the end time
                end_time = time.time()
                # Print the puzzle with hints and explanations
                print(f"Puzzle with {args.hint} hints:")
                print_puzzle(puzzle)
                print(f"Time taken: {end_time - start_time:.5f} seconds")

            elif not args.file:
                # If neither hint nor explain is specified, solve the puzzle
                if args.recursive:
                    solution = recursive_solve(puzzle, args.subgrid_rows, args.subgrid_cols, args.explain)
                elif args.wavefront:
                    solution = wavefront_solve(puzzle, args.subgrid_rows, args.subgrid_cols, args.explain)
                else:
                    solution = solve_puzzle(puzzle, args.subgrid_rows, args.subgrid_cols, args.explain)

                end_time = time.time()
                
                if solution:
                    # If a solution was found, print it and the time taken to find it.
                    print("Solved puzzle:")
                    print_puzzle(solution)
                    print(f"Time taken: {end_time - start_time:.5f} seconds")
                else:
                    # If no solution was found, print a message indicating that.
                    print("Could not solve puzzle.")
    
            # If a file was specified, attempt to read the puzzle from the file and solve it.
            if args.file:
                puzzle = read_puzzle(args.file[0])
                if args.recursive:
                    # If the recursive flag was set, attempt to solve recursively.
                    solution = recursive_solve(puzzle, args.subgrid_rows, args.subgrid_cols, args.explain)
                elif args.wavefront:
                    # If the wavefront flag was set, attempt to solve using the wavefront algorithm.
                    solution = wavefront_solve(puzzle, args.subgrid_rows, args.subgrid_cols, args.explain)
                else:
                    # Otherwise, solve using the default algorithm.
                    solution = solve_puzzle(puzzle, args.subgrid_rows, args.subgrid_cols, args.explain)
                end_time = time.time()

                if solution:
                    # If a solution was found, print it and the time taken to find it.
                    print("Solution:")
                    print_puzzle(solution)
                    print(f"Time taken: {end_time - start_time:.5f} seconds")
                    # If a solution file was specified, write the solution to the file.
                    if args.file[1]:
                        with open(args.file[1], 'w') as f:
                            f.write('\n'.join([', '.join([str(num) for num in row]) for row in solution]))
                else:
                    # If no solution was found, print a message indicating that.
                    print("No solution found.")
                    
    elif args.profile:
        # run the profiling code
        solver_performance(grids)

if __name__ == "__main__":
    main()
