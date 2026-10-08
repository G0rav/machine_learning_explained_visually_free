# How ML Models Work in Production — self-check

1. What happens when the user requests another penguin prediction?

   A. The loaded fitted pipeline processes the new measurements.
   B. Logistic regression is trained again using the new measurements.
   C. The scaler learns new statistics from that single bird.
   D. The user supplies the true species before inference.

2. What should this project's model artifact preserve?

   A. Only the Python source that constructs an unfitted classifier.
   B. Only the raw training measurements.
   C. Only the classifier, because numeric inputs need no preprocessing.
   D. The fitted scaler and trained classifier together.

3. A numeric input check accepts `4.98`. Does that establish correct units?

   A. Yes, because floating-point numbers always represent millimeters.
   B. No. The interface must specify units or perform an explicit conversion.
   C. Yes, because the scaler identifies centimeters automatically.
   D. No. Every prediction requires refitting the scaler first.

4. Which statement correctly separates inference timing from location?

   A. Batch inference requires a remote web API.
   B. On-device inference always runs as a scheduled batch job.
   C. A device can perform inference on demand or process a collection.
   D. Online, batch and on-device inference are mutually exclusive choices.

5. A prediction request succeeds. What can we conclude?

   A. The request produced a response; correctness needs a reliable outcome label.
   B. The predicted species is correct because the backend returned it.
   C. The incoming data distribution has stayed unchanged.
   D. Future predictions need no monitoring.

[Check your answers](solutions/quiz_answers.md).
