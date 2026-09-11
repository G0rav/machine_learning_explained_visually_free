# Machine Learning Explained Visually — Free Course

A free video course that teaches machine learning through visual,
step-by-step derivations instead of just equations and slides. Each video
covers one topic — an algorithm, or the mathematics an algorithm rests on —
and comes with a runnable Jupyter notebook that reproduces every number and
every chart shown on screen.

Nothing is assumed. Where a video needs mathematics you may not have, the
mathematics gets its own video rather than a hand-wave.

## Videos

| # | Video | Notebook | Length |
|---|-------|----------|--------|
| 01 | [**Linear Regression**](https://youtu.be/aRXqBKdyTWA) | [notebook](01-linear-regression/notebook.ipynb) · [open in Colab](https://colab.research.google.com/github/G0rav/machine_learning_explained_visually_free/blob/main/01-linear-regression/notebook.ipynb) | 41 min |
| 02 | [**Differentiation**](https://youtu.be/iRNatIA6hJc) — the calculus ML actually uses | [notebook](02-differentiation/notebook.ipynb) · [open in Colab](https://colab.research.google.com/github/G0rav/machine_learning_explained_visually_free/blob/main/02-differentiation/notebook.ipynb) | 30 min |
| 03 | [**Gradient Descent** — how models actually get trained](https://youtu.be/LHj_JsDbesg) | [notebook](03-gradient-descent/notebook.ipynb) · [open in Colab](https://colab.research.google.com/github/G0rav/machine_learning_explained_visually_free/blob/main/03-gradient-descent/notebook.ipynb) | 24 min |
| 04 | Gradient Descent for Linear Regression — *coming soon* | | |

## Running the notebooks

**Option 1 — Google Colab (no setup required)**

Click "open in Colab" next to any video above. The notebook runs in your
browser using Google's free hosted environment — nothing to install.

**Option 2 — Run locally**

```bash
pip install numpy pandas matplotlib scikit-learn sympy jupyter
jupyter notebook
```

Not every notebook needs all of those — video 2 requires only `numpy`, and says
so where it uses anything else — but that one line covers the whole course.

Each notebook ends with a self-check that asserts every figure shown in the
video. `checks passed` means your environment reproduces the video exactly.

## Why the numbers can be trusted

Nothing on screen is hand-typed. Every value in every animation is computed
from the same code that is in the notebook, and the notebook's assertions are
what keep the video and the code from drifting apart.

## Support the course / get notified of new lessons

This course is free. If you find it useful, the best way to
support it and get notified when new lessons are released is to subscribe to
the YouTube channel: **[https://www.youtube.com/@school_whool](https://www.youtube.com/@school_whool)**.

## Connect with the author

- YouTube: [https://www.youtube.com/@school_whool](https://www.youtube.com/@school_whool)
- LinkedIn: [linkedin.com/in/gaurav2022](https://www.linkedin.com/in/gaurav2022/)

## License

Code and notebooks: MIT (see [LICENSE](LICENSE)). Videos: © the author — free
to watch and link, not to re-upload.
