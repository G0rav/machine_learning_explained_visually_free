# ML in production — reference

| Term | Meaning in this project |
| --- | --- |
| Training | Learn parameter values from measurements and known species. |
| Inference | Use the fitted pipeline to predict from new measurements. |
| Pipeline | Connected preprocessing and prediction operations. |
| Model artifact | A saved fitted model or pipeline. |
| Frontend | Collect measurements and display the predicted species. |
| Backend | Receive, validate, prepare and process the input. |
| Web API | Define the destination, request and response for a program operation. |
| Endpoint | Address for an API operation, such as `/predict`. |
| Request body | Data supplied with the request. |
| Dependencies | Software packages needed to load and run the program. |
| Hosting | Provide an environment where the application runs. |
| Server process | Running program that accepts requests. |
| Online inference | Compute a prediction when requested. |
| Batch inference | Process a collection of inputs together. |
| On-device inference | Run the model on the device using it. |
| Latency | Elapsed time from request departure to response arrival. |
| Input drift | A change in the observed input pattern. |
| Rollback | Restore a previously reviewed model version. |

For the penguin pipeline, prepare `[[bill_length_mm, bill_depth_mm]]` in that
order. Preserve the fitted scaler with logistic regression. Load the artifact
before serving inputs. The reference input `[[49.8, 16.8]]` predicts Gentoo in
the reproduced first-project pipeline.

Check required fields and finite numeric values. Define units explicitly:
`4.98 cm × 10 = 49.8 mm`. A numeric validator cannot infer the user's unit.

Separate prediction from application action, service availability from model
correctness, and inference timing from inference location. Investigate drift
with data checks and reliable labels. Evaluate a candidate before releasing it;
an ordinary prediction request does not retrain the model.
