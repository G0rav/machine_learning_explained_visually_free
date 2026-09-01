# 01 · Linear Regression

**[Watch on YouTube](https://youtu.be/8CttUm53Y1I)** · 42 min

This lesson introduces linear regression using a small dataset of six used
cars. It builds up the model geometrically — as points, lines, and areas on a
chart — before writing any of it as an equation.

## What this video covers

The video works through a single running example — six used cars, predicting
the price of a seventh — and builds every concept from that example instead
of introducing it abstractly first.

**Setting up the problem**
- Regression vs. classification: predicting a continuous number
- Feature vs. target
- Why picking two points and drawing a line by hand gives a different answer
  every time depending on which points you pick

**A model, one piece at a time**
- The constant (mean) baseline, and why any real model has to beat it
- Residuals — the gap between actual and predicted price
- Reading the pattern in the residuals to see that age carries information
- The model equation, ŷ = wx + b: w (slope / coefficient / weight) tilts the
  line, b (intercept) moves it
- Why the intercept is needed — fitting with b forced to zero, and how much
  worse it scores
- Why residuals are measured vertically, not perpendicularly (age and price
  are different units — you can't measure across both at once)

**Scoring a line and training it**
- Why summing raw residuals fails (positive and negative cancel out)
- Squared error vs. absolute error, and why squaring is the usual choice
  (removes the sign, penalizes large misses harder, and is easier to
  differentiate)
- The objective function: SSE(w, b), written out term by term
- Ordinary least squares (OLS) — the model vs. the optimization method that
  fits it
- Parameter space: every (w, b) pair as a point, contours connecting pairs
  with equal score
- Gradient descent: estimating the gradient, stepping opposite it, and
  watching the score fall over hundreds of steps
- The learning rate, and a first look at what happens when it's too small or
  too large

**Reading the fitted model**
- Verifying the trained line by hand against every residual
- What the coefficient means, including its units (dollars per year vs.
  dollars per month — same line, different-looking number)
- What the intercept means, and why it can be an extrapolation itself
- Interpolation vs. extrapolation, and why predicting a car older than any in
  the dataset gives a negative, meaningless price

**Evaluating on a larger dataset**
- Moving from 6 cars to a noisier, 60-car simulated dataset
- RMSE — why squared error alone isn't interpretable, and how RMSE gets back
  to dollars
- R-squared, computed directly as 1 minus (model error / baseline error)
- Why a good training-set score doesn't tell you how the model performs on
  data it hasn't seen

**Where the model breaks**
- The linearity assumption, and checking it directly against the six cars
- Why an imperfect assumption doesn't automatically make the model useless

**In code**
- Implementing the objective function and gradient descent from scratch in
  NumPy, and confirming it lands on the same w and b as scikit-learn
- Fitting the same data with `scikit-learn`'s `LinearRegression`, including
  the 2D input-shape requirement that trips up most beginners

## Notebook

[`notebook.ipynb`](notebook.ipynb) reproduces every figure and number from the
video, and is organized in three parts:

| Part | Purpose |
|---|---|
| 1. Follow along | recomputes every number shown in the video |
| 2. Experiment | lets you modify the data (move a car, corrupt a row, change the learning rate) and see the effect |
| 3. Challenge | 49 assertions that must all pass — this is the self-check |

### Run it

**Google Colab (no install required):**
[open in Colab](https://colab.research.google.com/github/REPLACE_ME_USER/REPLACE_ME_REPO/blob/main/01-linear-regression/notebook.ipynb)

**Locally:**

```bash
pip install numpy pandas matplotlib scikit-learn jupyter
jupyter notebook notebook.ipynb
```

