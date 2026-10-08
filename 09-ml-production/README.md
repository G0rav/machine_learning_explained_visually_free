# How ML Models Work in Production — A Beginner's Guide

**[Watch on YouTube](https://youtu.be/0KbEA8C0s-8)** 

We already have a trained model. What needs to happen before an application can
use it? This guide follows the penguin classifier from the first ML project:
preserve the fitted pipeline, prepare inputs, compute a prediction, and return
the result. Then we separate inference timing, inference location, service
monitoring, and prediction quality.

The video explains production use. The notebook demonstrates the local code
behind those steps; it does not deploy a service.

| Material | Start here |
| --- | --- |
| Guided notebook | [notebook.ipynb](notebook.ipynb) |
| Input practice with answer checks | [practice.ipynb](practice.ipynb) |
| Worked practice | [solutions/practice.ipynb](solutions/practice.ipynb) |
| Technical reference | [cheat_sheet.md](cheat_sheet.md) |
| Written video companion | [lesson_notes.md](lesson_notes.md) |
| Application design challenge | [project.md](project.md) |
| Worked application design | [solutions/project_example.md](solutions/project_example.md) |
| Five-question knowledge check | [quiz.md](quiz.md) |
| Quiz explanations | [solutions/quiz_answers.md](solutions/quiz_answers.md) |
| Reusable local inference code | [production.py](production.py) |
| Data and attribution | [penguins.csv](data/penguins.csv) · [source](data/SOURCE.md) |

## Run the notebook

[Open in Colab](https://colab.research.google.com/github/G0rav/machine_learning_explained_visually_free/blob/main/09-ml-production/notebook.ipynb).
The link will work once this folder is published to the repository's main branch.
The notebook downloads its supporting code and dataset only if they are missing.

To run locally, use Python 3.10 or later and open a terminal in this folder:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Open `notebook.ipynb` and run the cells in order. Training creates a fitted scaler
and logistic regression model using the same training split as the first project.
Test rows remain unused. The notebook then writes its own artifact under
`artifacts/`, reloads it in the same environment, and predicts Gentoo from
`bill_length_mm=49.8` and `bill_depth_mm=16.8`. These are predictions, not verified
species labels. No prebuilt binary model or private video files are required.

## What you should be able to explain

- Why a prediction uses learned parameters rather than retraining the model.
- Why the fitted preprocessing and classifier must be preserved together.
- How named input fields become the ordered row expected by the pipeline.
- Why valid numeric inputs still need an explicit unit contract.
- How on-demand versus batch inference differs from device versus remote execution.
- Why a successful response does not establish a correct prediction.
- How labels, drift checks, reviewed releases, and rollback support ongoing use.

Start with the guided notebook, complete the practice, then write the application
design. Use the quiz before reading its answers. Code and notebooks use the root
repository's MIT license; the dataset has its own CC0 attribution.
