# 04 · Gradient Descent for Linear Regression

**[Watch on YouTube](https://youtu.be/eycqI9ifm54)** · 22 min

Video 3 built gradient descent on functions somebody chose. This one points it
at a real model, on real data — and then deals with the bill.

The bill is the half that usually goes unmentioned. The gradient of a loss is a
sum with one term per training row, and you need that gradient at *every*
update, at a different `w` each time, so nothing can be reused. **One honest
update is one full pass over your data.** On six cars that is free. On a million
rows, two thousand updates means two billion row visits before you have a model.
Stochastic gradient descent is the fix, and it is the algorithm that actually
trains everything you have heard of.

It picks up exactly where [video 3](../03-gradient-descent) left off, and you can
watch this one first — the update rule it assumes is stated in the first minute.

## What this video covers

**The loss function L(w), and why w is the variable**
- The rule from video 3, unchanged, with a loss `L` where `f` was and a weight
  vector `w` where the single number `x` was
- `L(w) = Σ (yᵢ − wᵀxᵢ)²`, with no regularisation and no intercept, so the
  algebra stays readable
- **The confusion that stops people**: what is the variable? The `x`s and the
  `y`s are your training data. They are given and they never move. The only
  thing you get to change is `w` — so you differentiate with respect to `w`
- Doing it: the outside is something squared, the chain rule wants the inside,
  and `∂(wᵀx)/∂w = x` from video 2. `∇L = Σ −2xᵢ(yᵢ − wᵀxᵢ)`
- Then it runs, on video 1's six cars. From a deliberately bad start the loss
  falls `134.98 → 63.27 → 58.37 → 46.77 → 21.51 → 4.61 → 4.48` over 2,000
  updates and lands on `w = −1.999995`, `b = 23.999967`
- The gradient at that bad start is `(246, 18)` — **the same two numbers video 1
  got by nudging each parameter and watching the total move.** Measured then,
  differentiated now

**The cost of one update: a full pass over the data**
- The summation runs over every row, and you need it again at every update
- Six rows: nothing. A million rows and two thousand updates: two billion row
  visits
- And it is not specific to linear regression — differentiate any loss that is
  a sum over your data and you get a gradient that is a sum over your data

**Stochastic gradient descent (SGD) and the batch size**
- The change is one character: don't sum over `n`, sum over `k`
- The condition that makes it work and is easy to get wrong: **you redraw the
  `k` rows every single iteration.** Pick a set once and keep it and you are
  just training on a smaller dataset
- Why it works, shown rather than asserted: the full gradient *is* the six
  per-row pieces added up, so a single row is one of the things being summed
- **And why that is not the whole story.** Those pieces disagree with each
  other. Two of the six sit 139° and 158° from the total — they point
  backwards, and a step along either one makes the loss go *up*. It still
  arrives, because thousands of freshly drawn steps average out to the
  direction the full gradient points
- The honest comparison, on data that shows it in a bad light: `k = 1` over
  20,000 updates reaches a loss of `4.70`, against full batch's `4.48` in 2,000.
  Ten times the iterations for a worse answer — because on six rows there was
  nothing to save. The argument is about the million
- Vocabulary: `k` is the batch size; `k = 1` is usually called stochastic,
  `1 < k ≪ n` is mini-batch, and 100 or 256 are common. People use the terms
  loosely — read the `k`

**Implementing SGD, and scikit-learn's SGDRegressor**
- The whole of video 3 in six lines of Python, stopping after 16 updates
- The same loop on the six cars: four lines of arithmetic inside it, and it
  reproduces the run from earlier in the video exactly
- `SGDRegressor` with every option set rather than defaulted, because the
  defaults quietly change the problem — the default penalty adds regularisation
  the loss does not have, and the default schedule changes `r` as it goes
- **The question thousands of people have asked and nobody has filmed the answer
  to**: it stops after 2,282 passes at `w = −1.995452`, which differs from our
  own loop in the third decimal place. That is not a bug in scikit-learn and it
  is not a bug in the loop. Near the minimum a single-row run does not settle
  onto a point — it orbits one, and where it stops is wherever it happened to be
  when the tolerance fired
- Push `eta0` from `0.002` to `0.02` and it comes back `−1.766528`. Still the
  right line, noticeably further from it. **With the stochastic version the
  learning rate is not only deciding how fast you arrive, it is deciding how
  close you stay**
- And `LinearRegression`, on the same six cars, returns `−2` and `24` to
  fourteen decimal places — because it is not iterating at all

**Summary, local minima, and solving directly instead**
- The three sentences worth keeping, and where each was derived
- **The limit, stated plainly.** Gradient descent only ever reads the derivative
  at the point it is standing on. It has no information about the function
  anywhere else. So on a function with more than one minimum, where you start
  decides which one you reach — two runs four tenths apart end up four whole
  units apart, and one of them finds the worse minimum
- The loss used here has exactly one minimum, which is why every run landed in
  the same place. That is a property of this loss, not a promise about the next
- And the thing the previous part left hanging: for linear regression
  specifically you can set the gradient to zero and *solve*. The loss is
  quadratic in `w`, so that equation is linear, so it has one answer and you can
  write it down

## Notebook

[`notebook.ipynb`](notebook.ipynb) reproduces every number in the video and is
organised in three layers:

| Layer | Purpose |
|---|---|
| 1. Follow along | every run in the video, in code, in the same order — 54 checked values |
| 2. Experiment | the whole batch-size line rather than the video's two points, what the bill looks like at five dataset sizes, the learning rate against the width of the orbit, and the basin map of a loss with two minima |
| 3. Challenge | four exercises, each with a check to run against your own answer |

Layer 1 prints `54 checks passed` when your environment reproduces the video
exactly. Every one of those values is asserted against the same code that drew
the animations, so the notebook and the video cannot drift apart quietly.

Three things in layer 2 are worth the detour even if you skip the rest.

**The crossover.** The video compares full batch against `k = 1` on six rows and
tells you that is the wrong dataset to judge it on. The notebook prices both at
five sizes, and the crossover sits around `n = 1,000`: below it full batch wins
outright, above it the gap is the reason every large model is trained the other
way.

**The orbit.** `SGDRegressor`'s answer differing in the third decimal is not
bad luck, it is the steady state of a single-row run. Sweep `eta0` from `0.0005`
to `0.05` and the *centre* of the run barely moves while the *spread* grows
roughly linearly — so `−1.7665` is not a worse answer, it is the same answer
sampled from a wider orbit. It is an ordinary draw from the wide one and an
impossible draw from the tight one, and the notebook says so in standard
deviations.

**The basin map.** Part 13 runs two starting points either side of a maximum.
The notebook runs twenty-seven, and the boundary between "finds the global
minimum" and "finds the worse one" falls exactly on a point the algorithm never
evaluates.

The notebook imports nothing from this repository. `numpy` is the only
requirement; `scikit-learn` and `matplotlib` are optional, and every cell that
uses either says so and keeps working without it — without scikit-learn the
library comparison falls back to the same fitted results, so the section still
states its numbers.

### Run it

**Google Colab (no install required):**
[open in Colab](https://colab.research.google.com/github/G0rav/machine_learning_explained_visually_free/blob/main/04-gradient-descent-linear-regression/notebook.ipynb)

**Locally:**

```bash
pip install numpy scikit-learn matplotlib jupyter
jupyter notebook notebook.ipynb
```

## Next

The answer this video spent two thousand updates walking to, written down in one
calculation — where `w = −2` and `b = 24` come from exactly, and why the
derivation needs nothing that videos 2 and 3 have not already given you.
