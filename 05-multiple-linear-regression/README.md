# 05 · Multiple Linear Regression

**[Watch on YouTube](https://youtu.be/lS16A11OiME)** · 31 min

Videos 1, 3 and 4 fitted a line through one feature. Add a second column and the
arithmetic barely changes — but the thing you were relying on quietly stops being
true.

**A coefficient in a model with several features is not a fact about the world.**
It is the effect of that one feature with the others held still, and if your
features move together, it is describing a situation that never actually happens.
Age costs `−1.2285` on its own. Put mileage beside it and the same age
coefficient becomes `−0.5043`, on the same sixty cars, with nothing about cars
having changed. This video is about why that happens, how to tell when it is
happening to you, and what to do about it.

It closes a four-video run on one algorithm, and you can watch it first — it
assumes nothing that is not restated in its first minute.

## What this video covers

**More than one feature**
- The equation with several columns, the compact `wᵀx` form, and the column of
  ones that carries the intercept
- One feature is a line, two is a plane, and beyond that a hyperplane you cannot
  draw — which is fine, because the algebra never needed the picture
- `np.linalg.solve(X.T @ X, X.T @ y)` on the sixty cars: `21.3373`, `−0.5043`,
  `−0.0614`
- **The coefficient moved.** Age was `−1.2285` alone and is `−0.5043` with
  mileage in the model. Same data, same target, different number — and neither
  is wrong

**You cannot compare them**
- Age is in years and mileage is in thousands of miles, so their coefficients
  are in different units and sorting them sorts the units, not the features
- Standardising — subtract the mean, divide by the standard deviation — puts
  them on one scale: `w_age −1.9844`, `w_mileage −2.9723`, `R² 0.9052`
- **The ranking flips the moment you do it.** The bigger raw coefficient is not
  the more important feature
- Only after standardising are the absolute values a ranking you can put in
  front of somebody

**Why the coefficients are arbitrary**
- Collinearity, defined: one feature is a constant times another, plus a
  constant. Here, `mileage = 12 × age + 0`
- The exact case, where three different models fit the six cars *identically* —
  not nearly, exactly — so there is a whole line of models and no reason to
  prefer any of them
- Real data is never exact, so instead of one answer you get a long thin valley
  in the loss surface, and the fit lands somewhere along it
- **The perturbation test**, which is the practical takeaway: shake the data
  slightly, fit again, and see whether the answers move. About four lines of
  code
- The swing, as correlation climbs: `0.0 → 0.0248`, `0.5 → 0.0303`,
  `0.9 → 0.0605`, `0.99 → 0.2223`, `0.999 → 0.3957`. The coefficients trade
  against each other while the predictions barely move

**Why the model memorises**
- Ten coin-flip columns are added — genuinely random, no relationship to price
- Ordinary least squares gives every one of them a weight: `coin9 +0.4639`,
  `coin7 +0.4319`, `coin10 +0.3525`. Against `age −1.5897` in that same fit, a
  coin flip carries nearly a third of the weight the age of the car does
- **And the score goes up**, `0.9052 → 0.9235`. That is arithmetic, not bad
  luck: a weight of zero reproduces the old model, so the best fit with a new
  column is never worse. R² on the training data rewards you for adding rubbish
- So eighteen of the sixty cars are held back. Two real features score
  `train 0.9014` / `test 0.8894` — level, and healthy
- The ladder as coins go in: `(0.9124, 0.8629)`, `(0.9399, 0.7775)`,
  `(0.9614, 0.5010)`, `(0.9758, 0.0623)` — and at 39 coins, `train 1.0000`
  against `test −6.3061`
- **What a negative R² means**: zero is what you score by predicting the average
  price for every car, so `−6.3061` is worse than never training anything
- The weights gave it away the whole time — the norm climbed `3.65 → 10.44`.
  Memorising needs big weights, **so make big weights expensive**

**The penalty term (ridge)**
- Add `λ‖w‖²` to the loss and you are asking for two things at once: fit the
  data, and stay small
- The path as `λ` rises — `0`, `1`, `10`, `50`, `100`, `500` — with `R²` falling
  `0.9052 → 0.3115` as the coefficients shrink
- It fixes the collinearity problem too: the long thin valley gets a floor, so
  the answer stops sliding along it
- **Not better, more repeatable.** Ridge trades a little fit for an answer that
  does not move when the data does

**The penalty that reaches zero (lasso)**
- Absolute value instead of squared, and the entire difference follows from one
  fact: **one derivative dies at zero and the other does not**
- So lasso sends coefficients to *exactly* zero and picks your features, while
  ridge shrinks everything and keeps it
- On the coins: `λ 0.2` leaves six alive, `λ 0.3` leaves four, and `λ 0.5`
  leaves exactly `age` and `mileage`
- The coefficient path, and where elastic net sits between the two

**One bad row**
- Residuals are your only eyes on a fit you cannot plot
- One wrong price does very different damage depending on where it sits: the
  same error at 13.4 years drags the slope to `−1.0006`, and at 5.0 years to
  `−1.3173`
- **Squared loss has no defence against it** — it is the one row furthest from
  the line, so it dominates the sum it is being fitted to
- Fit, look at the residuals, drop what is absurd, fit again — which is the idea
  behind RANSAC, and it works well beyond linear models

**Writing it**
- The design matrix, and `X^T X w = X^T y` — **you solve it, you do not invert
  it**, which is what every library actually does
- `make_pipeline(StandardScaler(), Ridge(alpha=10.0))`, with the scaling inside
  the pipeline so it cannot be forgotten. Your `λ` is the argument called `alpha`
- The four things to keep, and one deliberately left open: **how to choose `λ`**
  is a subject of its own

## Notebook

[`notebook.ipynb`](notebook.ipynb) reproduces every number in the video and is
organised in three layers:

| Layer | Purpose |
|---|---|
| 1. Follow along | every run in the video, in code, in the same order — 32 checked values |
| 2. Experiment | the age coefficient coming home, the swing against dialled correlation, the whole ridge path, and the lasso path with the point each coin dies |
| 3. Challenge | four exercises, each with a check to run against your own answer |

Layer 1 prints `32 checks passed` when your environment reproduces the video
exactly. Those values are asserted against the same code that drew the
animations, so the notebook and the video cannot drift apart quietly.

Three things in layer 2 are worth the detour even if you skip the rest.

**The coefficient coming home.** The video's central claim is that `−0.5043` was
never a fact about cars. Take mileage back out and watch age return to exactly
`−1.2285`. It is three lines, and running it yourself is the fastest way to stop
half-believing it.

**The swing, dialled.** The video shows the perturbation test at the correlation
this dataset happens to have. The notebook builds data at a correlation you
choose and sweeps it, so you can watch the swing grow from `0.0248` at zero
correlation to `0.3957` at `0.999` — and see that the predictions stay steady
the whole way. Unstable coefficients and a bad fit are different problems.

**Where each coin dies.** The video quotes three points on the lasso path. The
notebook walks it and prints the surviving set at each `λ`, so you can watch the
coins go out one at a time — `coin2` first at about `0.015`, `coin10` the last
of the ten to go at about `0.455`, and `age` and `mileage` still standing well
past `1.5`. Lasso picks the two real columns without being told which they were.

The notebook imports nothing from this repository and rebuilds the dataset from
its generator, so one downloaded file gives you the video's numbers. It needs
`numpy` and `scikit-learn` — the library `Ridge` and `Lasso` fits are imported
directly, with no fallback, so scikit-learn is required rather than optional.

### Run it

**Google Colab (no install required):**
[open in Colab](https://colab.research.google.com/github/G0rav/machine_learning_explained_visually_free/blob/main/05-multiple-linear-regression/notebook.ipynb)

**Locally:**

```bash
pip install numpy scikit-learn jupyter
jupyter notebook notebook.ipynb
```

## Next

This one closes the run on linear regression — what the model is, how it is
trained, and what goes wrong when it has more than one column to work with.

The thread deliberately left hanging is choosing `λ`. Picking it properly means
measuring on data the model has not seen, which is the same held-out idea this
video borrowed for one part and did not stop to teach.
