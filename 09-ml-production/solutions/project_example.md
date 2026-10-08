# Worked design: penguin measurement application

The application accepts bill length and bill depth in millimeters. It displays
the model's predicted species for a recorded observation. The interface labels
this as a prediction; a reliable observed species label remains separate.

## Input and prediction

The frontend sends a JSON object with `bill_length_mm` and `bill_depth_mm`.
The backend checks both fields, rejects unexpected fields, rejects strings,
booleans, missing measurements, infinity, NaN, and nonpositive measurements.
The frontend declares millimeters. If it accepts centimeters, it converts each
measurement explicitly before constructing the request. Numeric validation
cannot independently establish the measurement's unit or reliability.

The backend prepares `[[bill_length_mm, bill_depth_mm]]`. A process loads the
fitted `StandardScaler` and `LogisticRegression` pipeline before it accepts
requests. Each valid request calls `predict`, which transforms the ordered row
using the training statistics and applies the trained classifier. It does not
call `fit`. A response includes the predicted species and model version.

## Timing and location

For a field worker who needs a result immediately after recording a measurement,
choose on-demand inference. Remote inference lets a reviewed model version be
updated in one managed environment, but it requires network access and a running
backend. If the field location lacks a reliable connection, a compatible model
could instead run on the device and accept measurements there. This is a change
in location, not a requirement to use batch inference.

For a collection of observations that only needs results at the end of a day,
validate the records and predict them as a batch. Record the model version with
each result. Failed validation returns an input error rather than an invented
species. The frontend retains the entered measurements so they can be corrected.

## Checks after release

Track request errors, response latency, and process availability to assess
service operation. Separately compare predictions with species labels verified
by qualified observers to assess prediction quality. A valid response establishes
that the operation completed, not that its prediction is right.

Compare incoming feature distributions and missing-field rates with a documented
reference period. Investigate changed measurement procedures or populations.
Input drift alone does not prove an accuracy drop. If labels are delayed, report
that predictive correctness is not yet measured for those observations.

Train a candidate separately on reviewed data, compare it with the existing
version using an agreed evaluation process, and check that the saved pipeline
loads and accepts the same input contract. Release it as a new version only after
review. Keep the previous artifact and environment record so the serving process
can restore that version if the replacement fails. Ordinary requests never
initiate model training.
