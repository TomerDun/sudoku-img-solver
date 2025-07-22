import numpy as np


def in_arr(arr, num):
    for n in arr:
        if num == n:
            return True
    return False


def get_row(sud_table, sud_row, row):
    full_row = sud_table[sud_row, :, row, :]
    full_row = full_row.flatten()
    return full_row


def get_column(sud_table, sud_col, column):
    full_column = sud_table[:, sud_col, :, column]
    full_column = full_column.flatten()
    return full_column


def print_all(sud_table):
    text = ""
    for i in range(len(sud_table)):
        # Do This the number of sud_rows (3)
        for u in range(len(sud_table[i])):
            # Do this the number of tables in the row (3)
            full_row = get_row(sud_table, i, u)
            # Print The Row with spacing
            for num in range(1, len(full_row) + 1):
                text += str(full_row[num - 1])
                if num % 3 == 0:
                    text += "  "
            text += "\n"
        text += "\n"

    text = text.replace('0', 'X')
    print(text)


def generate():
    sud_table = np.zeros((3, 3, 3, 3), dtype="int32")
    for sud_row in sud_table:
        for sud in sud_row:
            for row in sud:
                for i in range(len(row)):
                    row[i] = np.random.randint(0, 10)
                    #row[i] = 0

    return sud_table


def solve_once(old_sud_table):
    sud_table = old_sud_table.copy()
    # MAIN ALGORITHMS
    for sud_row_num, sud_row in enumerate(sud_table):  # For Each Sudoku Row
        for col_num, sud in enumerate(sud_row):  # For Each Sudoku In The Row
            for row_num, row in enumerate(sud):  # For Each Row Of Numbers In The Sudoku
                for col, spot in enumerate(row):  # For Each Spot

                    if spot == 0:
                        # Solve This Spot
                        sol_count = 0
                        solution = 0
                        for i in range(1, 10):
                            if not in_arr(get_row(sud_table, sud_row_num, row_num), i) and not in_arr(get_column(sud_table, col_num, col), i) and not in_arr(sud.flatten(), i):
                                # If found a possible solution for this spot
                                sol_count += 1
                                solution = i
                        if sol_count == 1:
                            # If there is only one possible solution for this spot
                            row[col] = solution

                    if row[col] == 0:  # If Unable to Solve
                        row[col] = 0
                        # return -1
    return sud_table


def full_solve(sud_table):
    solved_table = sud_table.copy()

    for i in range(len(sud_table.flatten())): # Do this times the spots in the sudoku table
        solved_table = solve_once(solved_table)

    return solved_table

def empty_sudoku():
    sudoku_table = np.array([
        [
            [[5, 3, 0], [6, 0, 0], [0, 9, 8]],
            [[0, 7, 0], [1, 9, 5], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 6, 0]]
        ],
        [
            [[8, 0, 0], [4, 0, 0], [7, 0, 0]],
            [[0, 6, 0], [8, 0, 3], [0, 2, 0]],
            [[0, 0, 3], [0, 0, 1], [0, 0, 6]]
        ],
        [
            [[0, 6, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [4, 1, 9], [0, 8, 0]],
            [[2, 8, 0], [0, 0, 5], [0, 7, 9]]
        ]
    ], dtype="int32")
    return sudoku_table

'''
sudoku_table = np.array([
    [
        [[5,3,0], [6,0,0], [0,9,8]],
        [[0,7,0], [1,9,5], [0,0,0]],
        [[0,0,0], [0,0,0], [0,6,0]]
    ],
    [
        [[8,0,0],[4,0,0] ,[7,0,0]],
        [[0,6,0], [8,0,3], [0,2,0]],
        [[0,0,3], [0,0,1], [0,0,6]]
    ],
    [
        [[0,6,0], [0,0,0], [0,0,0]],
        [[0,0,0], [4,1,9], [0,8,0]],
        [[2,8,0], [0,0,5], [0,7,9]]
    ]
], dtype="int32")

print_all(sudoku_table)

print("\n\n")

solved_sudoku = full_solve(sudoku_table)

print_all(solved_sudoku)
'''