# Computer Vision (CAP 5410)

Coursework for CAP 5410, Computer Vision, Florida Polytechnic University.
Each assignment lives in its own folder with its code and a
`requirements.txt`. Only the code is published here; reports and generated
results are kept out of the repo.

| Folder | Topic | Stack |
| --- | --- | --- |
| [`assignment1`](assignment1) | Video brightness/contrast editor with live histograms | OpenCV, Tkinter |
| [`assignment2`](assignment2) | Box, Gaussian and Sobel filters, manual vs. OpenCV | NumPy, OpenCV, Jupyter, pytest |
| [`assignment3`](assignment3) | LeNet-5 style CNN on MNIST, learning-rate comparison | PyTorch, scikit-learn |

## Setup

Each assignment has its own virtual environment:

```
cd assignmentN
python -m venv env
env\Scripts\activate            # Windows
source env/bin/activate         # Linux / macOS
pip install -r requirements.txt
```

Assignment 3 downloads MNIST into `assignment3/data/` on first run, so the
dataset is not stored in the repo. Running `python main.py` there recreates
`results/` (histories, weights, metrics) and the figures.
