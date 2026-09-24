# 06 · Class Imbalance — 98% accuracy, 0% fraud caught

**[Watch on YouTube](https://youtu.be/ZVEzPWNNPeg)** · 19 min 

A classifier that flags every transaction as legitimate scores 98% accuracy on
5,000 real-shaped transactions — and catches zero of the 100 fraud cases. This
lesson diagnoses that failure and fixes only what is actually broken: not the
model family, the metric.

It is a standalone special rather than part of the main sequence, so it does
not assume any earlier video. One binary fraud-detection dataset carries the
whole argument from a raw target count to a single, sealed test-set result.

## What this video covers

**The majority baseline, and why accuracy hides the failure**
- 5,000 transactions, 100 fraud — predict "legitimate" for every row and
  accuracy is `4,900 / 5,000 = 98%`
- Fraud recall on that same baseline is `0 / 100 = 0%`
- Each validation row contributes equally to accuracy, so missing every
  fraud case only costs two percentage points

**Classifier scores and the decision threshold**
- A score is not a class — a threshold turns it into one
- Moving the threshold from 0.500 to 0.300 flips a prediction while the score,
  and its ranking, never change
- Selecting a threshold does not retrain the classifier

**Class weighting**
- Each training row is weighted by the inverse of its class frequency —
  fraud rows count 25x more during fitting, and no row is duplicated or removed

**Random resampling and SMOTE**
- Random oversampling duplicates minority rows until the classes are equal
- Random undersampling removes majority rows until the classes are equal
- SMOTE interpolates a synthetic minority row between a real one and one of
  its nearest minority neighbors
- All three touch only the training partition — validation and test rows are
  never resampled

**Precision, recall, threshold selection — and the verdict**
- Missing a fraud transaction costs 100 units here; a false alarm costs 5, so
  operational cost is `100 · FN + 5 · FP`
- Sweeping the threshold on validation data drops cost from 1,705 at the
  default 0.5 to 420 at `t = 0.084`
- The precision-recall curve gives an average precision of 0.469
- All five candidates — threshold tuning alone, class weighting, oversampling,
  undersampling, SMOTE — detect 18 of 20 validation fraud rows; weighting and
  resampling only add more false positives here, so **threshold tuning alone
  wins on validation cost**
- Evaluated once on the sealed test set: fraud recall rises from 0% to 65%,
  and cost falls from 2,000 to 865

**Leakage-safe model selection**
- Resampling before splitting the data can put copies of the same row on both
  sides of the split — split first, resample only the training fold
- The threshold must be selected from validation data, never from test
  outcomes
- Every candidate needs its own threshold selected the same way — comparing
  four models at a shared default of 0.5 is not a fair comparison

## Notebook

[`notebook.ipynb`](notebook.ipynb) reproduces every number in the video and is
organized in three layers:

| Layer | Purpose |
|---|---|
| 1. Follow along | every derivation in the video, in code, in the same order — 32 checked values |
| 2. Experiment | the video's own closing challenge: change the fraud prevalence and the false-negative cost, and watch the selected threshold — and the winning intervention — move |
| 3. Challenge | write your own cost-based threshold selector and your own SMOTE interpolation step, then check them |

Layer 1 prints `32 checks passed` when your environment reproduces the video
exactly. Every value — the dataset counts, the fitted logistic regression, the
selected threshold, the five candidates' validation costs, the final test
result — is asserted against the same computation that produced the numbers on
screen, so the notebook and the video cannot drift apart quietly.

Layer 2 is worth the detour even if you stop there. At a false-negative cost of
20, class weighting overtakes threshold tuning as the winner; at 500, random
oversampling does. The dataset never changes — only the price of a mistake
does, and that alone is enough to change which intervention you should ship.

The notebook imports nothing from this repository — `numpy` is the only
requirement.

### Run it

**Google Colab (no install required):**
[open in Colab](https://colab.research.google.com/github/G0rav/machine_learning_explained_visually_free/blob/main/06-class-imbalance/notebook.ipynb)

**Locally:**

```bash
pip install numpy jupyter
jupyter notebook notebook.ipynb
```
