# sudoku-img-solver

Lightweight Sudoku-from-image solver using Python, OpenCV & NumPy.

## 📝 Overview

Detects a Sudoku grid in an image, recognizes digits via template matching, solves the puzzle with a backtracking algorithm, and overlays the solution back onto the original image.
The program will show a step-by-step solution. solving each cell. when an image is shown, press any key to continue to the next cell.

## ⚙️ Installation

```bash
# 1. Create & activate a virtual environment
python3 -m venv venv
# macOS/Linux
source venv/bin/activate
# Windows
# venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

1. Place your puzzle image at `Sudokus/sud1.png` (or update the path in `SudLearner.py`).  
2. Run the solver:
   ```bash
   python SudLearner.py
   ```
3. The script will:
   - Show intermediate detection/overlay steps in OpenCV windows  
   - Print the unsolved and solved grids to your terminal  

## 📦 As a Module

You can also import and use the solver functions in your own code:

```python
from sudoku_solver import empty_sudoku, full_solve, print_all

grid  = empty_sudoku()     # 9×9 grid filled with zeros
solved = full_solve(grid)  # returns a solved 9×9 array
print_all(solved)          # pretty-print to console
```

## 📂 Project Structure

```
sudoku-img-solver/
├── requirements.txt      # opencv-python, numpy
├── SudLearner.py         # main script: detection → OCR → solve → overlay
├── sudoku_solver.py      # solver utilities & backtracking algorithm
└── Sudokus/              # sample puzzles & digit-template assets
    ├── sud1.png
    └── Templates/
        ├── t_empty.png
        ├── t_1.png
        └── …  
```
