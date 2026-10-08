# 08 · Principal Component Analysis

**[Watch on YouTube](https://youtu.be/7dCs1PjjTdg)** 

This notebook follows the video's PCA derivation on four customers, then runs
the same method on 500 **synthetic** customer rows. All data is generated inside
the notebook with a fixed seed. No CSV download or private course code is
needed. The coupon outcome is never used to fit PCA.

## Start with the notebook

[`notebook.ipynb`](notebook.ipynb) has three layers:

| Layer | What you do |
| --- | --- |
| Follow the video | Center and project four rows, compare NumPy with scikit-learn on the same rows, reconstruct a new customer, standardise four features, and check explained variance, reconstruction error, and coupon prediction. |
| Experiment | Compare raw and standardised PCA with one, two, or three components on identical training and held-out rows. |
| Your turn | Answer six small numerical and coding tasks. Each check prints ✅, ❌ with the expected value, or ⬜ when unanswered. |

An untouched notebook runs to the end with six ⬜ checks. The separate
[`solutions/notebook.ipynb`](solutions/notebook.ipynb) runs with six ✅ checks
and includes reference explanations. Try the learner copy first.

The PCA directions returned by NumPy or scikit-learn may point opposite to the
ones drawn in the video. The worked cells align their signs before comparing
scores and explain why both directions describe the same component line.

### Run it

**Google Colab:** [open the learner notebook in Colab](https://colab.research.google.com/github/G0rav/machine_learning_explained_visually_free/blob/main/08-principal-component-analysis/notebook.ipynb). The link will work after this folder is published to the repository's `main` branch.

**Locally:** use Python 3.10 or newer, then run:

```bash
python3 -m pip install numpy scikit-learn jupyter
python3 -m jupyter notebook notebook.ipynb
```

Choose **Restart Kernel and Run All Cells**. The notebook was verified from
outside the course repository with NumPy 1.26.4 and scikit-learn 1.7.2. Its
saved outputs show the video values; running it yourself checks them again.

The 55%, 92%, and 91% coupon accuracies are results for this generated dataset
and its fixed split, not estimates of real customer behaviour.
