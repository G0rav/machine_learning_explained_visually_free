# 07 · Build Your First Machine Learning Project You Can Actually Explain

Video link: to be added when the video is published.

- We use only bill length and bill depth as predictors so every row can be plotted in two dimensions. `island`, `sex`, `year`, flipper length, and body mass are excluded. The project predicts the three `species` labels.

This folder is the download for the first ML project video. It carries the
original Palmer Penguins CSV through one complete beginner workflow: inspect
the data, keep two bill measurements, remove two incomplete rows, reserve a
sealed test set, establish a baseline, fit logistic regression, and inspect
every test error.

The model gets 66 of 69 held-out birds right (95.7%). The always-Adelie
baseline gets 30 of 69 right (43.5%). These are results on this one test split,
not a claim about penguins in other populations.

## Start with the notebook

[`notebook.ipynb`](notebook.ipynb) follows the 14 code sections shown in the
video. It includes saved outputs, a setup cell for Google Colab, and a final
12-check self-test. The CSV and project script are also included so the
notebook runs locally without downloading anything.

**Google Colab:** [open the notebook in Colab](https://colab.research.google.com/github/G0rav/machine_learning_explained_visually_free/blob/main/07-first-ml-project/notebook.ipynb). The link will work after this folder is published to the repository's `main` branch. Run the cells in order; the first setup cell downloads the supporting files into Colab and checks the CSV fingerprint.

**Locally:** from this folder, with Python 3.10 or newer:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt jupyterlab ipykernel
.venv/bin/python -m jupyter lab
```

Open `notebook.ipynb` and choose **Restart Kernel and Run All Cells**. Start
JupyterLab from this folder so it finds `data/penguins.csv` and `project.py`.
The last cell prints `12 checks passed` when the dataset and results match the
video.

You can reproduce the complete project without Jupyter:

```bash
.venv/bin/python project.py
```

That command refreshes `results.json` with the split, predictions, confusion
matrix, and three mistaken birds.

## Files

| File | Purpose |
| --- | --- |
| [`notebook.ipynb`](notebook.ipynb) | Run the project in the order shown on screen. |
| [`data/penguins.csv`](data/penguins.csv) | Unmodified Palmer Penguins source data. |
| [`data/SOURCE.md`](data/SOURCE.md) | Dataset origin, CC0 license, fingerprint, and scope. |
| [`project.py`](project.py) | Standalone implementation of the same calculation. |
| [`results.json`](results.json) | Saved detailed results from the project script. |
| [`requirements.txt`](requirements.txt) | NumPy and scikit-learn requirements. |

Only bill length and bill depth are used as features. The model's 95.7% test
accuracy does not establish performance on new populations or field conditions.
