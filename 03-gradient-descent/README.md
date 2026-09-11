# 03 · Gradient Descent — how machine learning models actually get trained

**[Watch on YouTube](https://youtu.be/LHj_JsDbesg)** · 24 min

Every model you have ever trained was trained by one line of arithmetic, run
over and over. This lesson builds that line from nothing and then spends half
its time on the two ways it quietly fails — which is the half most
explanations skip.

There is no dataset and no model here. Everything is developed on a generic
function, so the algorithm is never hiding behind an example, and there is no
ball and no hill: the picture in this lesson is the update rule itself, not a
metaphor for it.

It picks up where [video 2](../02-differentiation) left off. That video ended
on an equation that cannot be rearranged — you can differentiate it and set the
derivative to zero, and then there is simply no way to isolate `x`. This one is
about what you do instead.

## What this video covers

**Why gradient descent is an iterative algorithm**
- The method from video 2: differentiate, set to zero, solve — and when it works
- The regularised logistic loss, where it does not, and why "no closed form" is
  a statement about algebra rather than about the answer
- What iterating means: guess, improve, repeat, stop
- Minimising `f` is maximising `-f`, so everything is written for minima only

**The sign of the derivative around a minimum**
- Positive on one side, negative on the other, zero at the bottom
- So the sign already says which way to step, and nothing ever has to ask
- And the *size* shrinks as you come in — `3`, `1.2`, `0.48` — which part 5 turns
  into the most useful property the algorithm has

**The gradient descent update rule**
- `x₁ = x₀ − r · f'(x₀)`, applied by hand from `3`, from `0`, and from `−1`
- The same expression, with the same sign in it, moves left from one side and
  right from the other — with no `if` statement anywhere

**The iteration loop, stopping criterion and tolerance**
- The rule with an index on it, run until it stops moving
- Why the test is on the movement and not on the derivative: with real
  arithmetic the derivative essentially never reaches exactly zero
- What a tolerance costs: `1e-4`, `1e-6` and `1e-9` take 11, 16 and 24 updates
- Thirteen extra updates for five extra digits

**Why the step size shrinks automatically**
- The steps taken were `0.9`, `0.36`, `0.144`, `0.0576`, `0.02304`
- `r` never changed — nothing in the algorithm says "take shorter steps later"
- The step is `r` times the derivative, and the derivative is the distance to
  the minimum doubled, so each step leaves exactly `1 − 2r` of that distance
- Far out it covers ground; close in it settles without overshooting, from one
  fixed constant

**The learning rate hyperparameter, and too small**
- `r` is the learning rate, also called the step size, written `eta` in papers
- Nothing computes it — you supply it — which is what makes it a hyperparameter
- At `r = 0.02` every step is correct and the loss falls every time, and it
  still takes 123 updates to do what `r = 0.3` does in 6
- **The failure that looks healthy**: a loss still coming down looks exactly
  like a run that is working, because it *is* one. What you cannot see is
  whether it needs another hundred updates or another ten million

**Oscillation, divergence and learning rate schedules**
- At `r = 1` it lands the same distance out the other side, forever: `0`, `3`,
  `0`, `3`, at a loss that never moves
- At `r = 1.05` the steps grow and it leaves — so a maximum-iteration cap is
  not belt and braces, it is the only exit
- The fix: stop keeping `r` constant. Make it a function of the iteration
  number, `r = h(i)`, decreasing

**Gradient vectors, and a loss with no closed form**
- In machine learning `x` is a vector, and `df/dx` becomes `grad f` — the
  vector of partial derivatives, one per component
- That is the whole change: the subtraction happens component by component and
  the algorithm does not know there is more than one dimension
- Then the function from part 1, the one that will not solve: eleven updates at
  `r = 2` land on `1.177505`, a minimum that cannot be written down

## Notebook

[`notebook.ipynb`](notebook.ipynb) reproduces every number in the video and is
organized in three layers:

| Layer | Purpose |
|---|---|
| 1. Follow along | every run in the video, in code, in the same order — 85 checked values |
| 2. Experiment | the whole learning-rate line rather than the video's four points, what a tolerance costs per decimal place, and the floating-point floor |
| 3. Challenge | four exercises, each with a check to run against your own answer |

Layer 1 prints `85 checks passed` when your environment reproduces the video
exactly. Every one of those values is asserted against the same code that drew
the animations, so the notebook and the video cannot drift apart quietly.

Two things in layer 2 are worth the detour even if you skip the rest. Part 5
derives that each update leaves `1 − 2r` of the distance to the minimum; that
same number answers *does this converge at all?*, because the gap shrinks only
while `|1 − 2r| < 1`, which is `0 < r < 1`. The boundary is not a rule of
thumb, it is arithmetic — and `r = 0.5` makes the factor exactly zero, so it
lands on the answer in a single step.

The other is the floating-point floor. The video says a program cannot wait for
the derivative to reach exactly zero because *with real arithmetic it usually
never will*. That "usually" is doing real work: on this curve at `r = 0.3` it
does reach exactly zero, at update 41 — and at `r = 0.1`, same curve, it never
does. You cannot tell which you have until you have already run it, which is
the whole reason the termination test is on the movement.

The notebook imports nothing from this repository — `numpy` is the only
requirement, and `matplotlib` is optional, with the one cell that uses it
guarded and labelled.

### Run it

**Google Colab (no install required):**
[open in Colab](https://colab.research.google.com/github/G0rav/machine_learning_explained_visually_free/blob/main/03-gradient-descent/notebook.ipynb)

**Locally:**

```bash
pip install numpy matplotlib jupyter
jupyter notebook notebook.ipynb
```

## Next

Video 4 points this algorithm at a real model: the loss becomes a sum over your
training data, the gradient becomes a sum with one term per row — so one honest
update reads the whole dataset — and stochastic gradient descent is what makes
that affordable.
