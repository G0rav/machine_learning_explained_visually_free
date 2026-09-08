# 02 · Differentiation — the calculus machine learning actually uses

**[Watch on YouTube](https://youtu.be/iRNatIA6hJc)** · 30 min

This lesson builds the derivative from nothing and stops exactly where machine
learning stops needing it. There is no integration, no trigonometry beyond one
angle, and no dataset: every idea is developed on a generic function, so the
mathematics is never hiding behind an example.

It picks up where [video 1](../01-linear-regression) left off. That video
trained a line with gradient descent and found the direction by nudging each
parameter and measuring what happened. This one is about the exact alternative
to that measurement.

## What this video covers

**Why derivatives run machine learning**
- What gradient descent did in video 1, and what part of it was an estimate
- The cost of nudge-and-measure: one extra function evaluation *per parameter*,
  which is why it does not survive contact with a real model
- The derivative as the exact answer to the same question

**What a derivative is measuring**
- Change in y over change in x, between two points on a curve
- The secant line, and Δ as nothing more than "the difference"
- Why that ratio is `tan(theta)` — the slope and the angle are the same fact

**Letting the gap go to zero**
- Walking the second point in, and watching the ratio settle
- Secant becoming tangent, and the limit definition of the derivative
- Why nothing is ever divided by zero: you divide by `h` while `h` is nonzero,
  and only then let it go

**Reading the sign**
- Positive, negative and zero slope, read off a curve before computing anything
- The peak of a hump as the place where the derivative is zero
- Which way a step should go, given the sign

**The rules**
- `d/dx x^2` derived from the definition, in full, as the one worked limit
- The power rule `x^n -> n x^(n-1)`, for any integer n, positive or negative
- A constant differentiates to zero, and a constant multiplier comes along
- `log x -> 1/x`, and `e^x -> e^x` — the one function that is its own derivative
- The sum rule, so a sum can be split up and done a piece at a time

**The chain rule**
- Functions inside functions, and why this is the rule neural networks are made of
- `(a - bx)^2` differentiated by the chain rule, then checked against expanding
  the bracket first and differentiating term by term

**Maxima and minima**
- Stationary points: where the derivative is zero
- The turning points of a cubic, found exactly

**Actually finding one**
- `f(x) = x^2 - 3x + 2`: differentiate, set to zero, solve, get `x* = 1.5`
- Checking *both* sides — one point higher and one point lower — and why one side
  is not enough (`x^3` at zero is neither a maximum nor a minimum)

**Local and global**
- A curve with two dips, three stationary points, and two different answers
- Why "the derivative is zero" does not mean "this is the best point"

**When you can't solve it**
- The softplus `log(1 + e^x)`, differentiated with the chain rule
- Setting the derivative to zero and working the equation down to `e^(ax) = 0` —
  and why that has no solution, for any a
- So the equation cannot always be solved, which is the entire reason iterative
  methods like gradient descent exist

**When x is a vector**
- Partial derivatives: differentiate with respect to one component, treat the
  rest as constants
- The gradient `grad f`, one entry per component of x, and why the notation
  itself tells you whether x is a scalar or a vector
- `grad(a^T x) = a`, computed one partial at a time

## Notebook

[`notebook.ipynb`](notebook.ipynb) reproduces every number in the video and is
organized in three layers:

| Layer | Purpose |
|---|---|
| 1. Follow along | every derivation in the video, in code, in the same order — 50 checked values |
| 2. Experiment | break the results on purpose: the step size that is too small, the neighbour-check that gets it wrong, and the cost of nudging across a model's worth of parameters |
| 3. Challenge | four exercises, each with a check to run against your own answer |

Layer 1 prints `50 checks passed` when your environment reproduces the video
exactly. Every one of those values is asserted against the same code that drew
the animations, so the notebook and the video cannot drift apart quietly.

The notebook imports nothing from this repository — `numpy` is the only
requirement, and `sympy` and `matplotlib` are optional, with every cell that
uses them guarded and labelled.

### Run it

**Google Colab (no install required):**
[open in Colab](https://colab.research.google.com/github/G0rav/machine_learning_explained_visually_free/blob/main/02-differentiation/notebook.ipynb)

**Locally:**

```bash
pip install numpy sympy matplotlib jupyter
jupyter notebook notebook.ipynb
```
