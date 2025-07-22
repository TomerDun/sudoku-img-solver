import cv2
import numpy as np
import sudoku_solver


class Cell:
    def __init__(self, img, number, spot, tl, br):
        self.img = img
        self.height, self.width = self.img.shape
        self.number = number
        self.spot = spot
        self.top_left = tl
        self.bot_right = br


class Matrix:
    def __init__(self, img, tl, br):
        self.img = img
        self.top_left = tl
        self.bot_right = br
        self.cells = [[]]
        self.spot = None

    def set_cells(self, cell_mat):
        self.cells = cell_mat

    def discover(self):  # Get the numbers from the image and sort them in a 3,3 array
        cells = classify_numbers(self.img)
        cells = clean_dups(cells)
        cells = sort_locations(cells)
        self.cells = cells


def classify_numbers(img):
    cells_arr = []
    for num, template in enumerate(num_templates):
        h, w = template.shape
        result = cv2.matchTemplate(img, template, method=method)
        threshold = 0.999

        # Locate Template in the image
        locations = np.where(result >= threshold)
        for top_left in zip(*locations[::-1]): # For Each num found
            bot_right = (top_left[0] + w, top_left[1] + h)
            # Add a cell
            cell = Cell(template, num, img[top_left[1]:bot_right[1], top_left[0]: bot_right[0]], top_left, bot_right)
            cells_arr.append(cell)
    return cells_arr


def classify_suds(img):  # Classify the tables in image and Return 1D array of Matrixs
    mat_arr = []

    template = sud_template
    h, w = template.shape
    result = cv2.matchTemplate(img, template, method=method)
    threshold = 0.972

    # Locate Template in the image
    locations = np.where(result >= threshold)
    for top_left in zip(*locations[::-1]):  # For Each matrix found
        bot_right = (top_left[0] + w, top_left[1] + h)
        # Create the Matrix
        matrix = Matrix(img[top_left[1]:bot_right[1],top_left[0]:bot_right[0]], top_left, bot_right)
        mat_arr.append(matrix)
    return mat_arr


def clean_dups(arr):
    threshold = 5
    black_list = []

    for image_i, image in enumerate(arr):
        if image_i in black_list:
            pass # If this was already removed pass
        else:
            for img_i, img in enumerate(arr):
                if img_i != image_i:  # Check if this is the same image
                    # Check the distance
                    if abs(img.top_left[0] - image.top_left[0]) < threshold:
                        if abs(img.top_left[1] - image.top_left[1]) < threshold:
                            black_list.append(img_i)
    # Create new array without the blacklisted items
    new_arr = arr.copy()
    for i in range(len(arr)):
        if i in black_list:
            new_arr.remove(arr[i])
    return new_arr


def sort_locations(arr):
    # sorted_arr is arr sorted by Y of top_left
    sorted_arr = bsort_tl(arr, 1)
    matrix = [
        bsort_tl(sorted_arr[0:3], 0),
        bsort_tl(sorted_arr[3:6], 0),
        bsort_tl(sorted_arr[6:9], 0)
    ]
    return matrix


def bsort_tl(arr, dim):
    A = arr.copy()
    for index in range(0, len(A) - 1):
        for j in range(0, len(A) - index - 1):
            if A[j].top_left[dim] > A[j + 1].top_left[dim]:
                A[j], A[j + 1] = A[j + 1], A[j]
    return A


def make_sudoku(suds_arr):
    # Get a 3,3 arr of Matrixs and Return a 3,3 arr of numbers
    sudoku = sudoku_solver.empty_sudoku()

    for mr_i, mat_row in enumerate(suds_arr):  # For each row of Matrixs
        for m_i, matrix in enumerate(mat_row):  # For each Matrix Object
            for cr_i, cell_row in enumerate(matrix.cells):  # For each row of numbers in the matrix
                for c_i, cell in enumerate(cell_row):  # FOR EACH CELL
                    sudoku[mr_i][m_i][cr_i][c_i] = cell.number

    return sudoku


def image_solve(old_sud, solved_arr, image):
    solved_sud = old_sud.copy()
    new_image = image.copy()

    for mr_i, mat_row in enumerate(solved_sud):  # For each row of Matrixs
        for m_i, matrix in enumerate(mat_row):  # For each Matrix Object
            mat_tl = matrix.top_left
            for cr_i, cell_row in enumerate(matrix.cells):  # For each row of numbers in the matrix
                for c_i, cell in enumerate(cell_row):  # FOR EACH CELL
                    if cell.number == 0:
                        # Change the image prop
                        img_temp = solved_sud[mr_i][m_i].cells[cr_i][c_i].img = num_templates[solved_arr[mr_i][m_i][cr_i][c_i]]
                        # Get the mask location
                        mask_loc = (
                            (cell.top_left[0] + template_w) + mat_tl[0] + padding[0],
                            (cell.top_left[1] + template_h) + mat_tl[1] + padding[1]
                        )
                        # Apply the mask
                        new_image[cell.top_left[1] + mat_tl[1] + padding[1]:mask_loc[1], cell.top_left[0] + mat_tl[0] + padding[0]:mask_loc[0]] = img_temp

                        # Show the Process
                        cv2.imshow(f'sr:{mr_i},s:{m_i},r:{cr_i},col:{c_i}', new_image)
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()

    return solved_sud


# Image Loading and variable Creation
img_full = cv2.imread('Sudokus/sud1.png', 0)
num_templates = [
    cv2.imread('Sudokus/Templates/t_empty.png', 0),
    cv2.imread('Sudokus/Templates/t_1.png', 0),
    cv2.imread('Sudokus/Templates/t_2.png', 0),
    cv2.imread('Sudokus/Templates/t_3.png', 0),
    cv2.imread('Sudokus/Templates/t_4.png', 0),
    cv2.imread('Sudokus/Templates/t_5.png', 0),
    cv2.imread('Sudokus/Templates/t_6.png', 0),
    cv2.imread('Sudokus/Templates/t_7.png', 0),
    cv2.imread('Sudokus/Templates/t_8.png', 0),
    cv2.imread('Sudokus/Templates/t_9.png', 0),

]

template_h, template_w = num_templates[1].shape
padding = (template_h//2, template_w//2)
sud_template = cv2.imread('Sudokus/Templates/t_matrix_blank.png', 0)

methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]
method = methods[3]

# Main

# Classify suds in the image
full_matrix = classify_suds(img_full)
full_matrix = clean_dups(full_matrix)
full_matrix = sort_locations(full_matrix)

# Sort numbers in each sud
for row in full_matrix:
    for s in row:
        # For each Matrix object in each row of the Sudoku
        s.discover()  # Sort the numbers in the Matrix

sudoku_numbs = make_sudoku(full_matrix)
sudoku_solver.print_all(sudoku_numbs)
solved = sudoku_solver.full_solve(sudoku_numbs)
print("------------------")
sudoku_solver.print_all(solved)

solved_matrix = image_solve(full_matrix, solved, img_full)














