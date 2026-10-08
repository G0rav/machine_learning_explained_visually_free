# How ML Models Work in Production

Written companion to the video.

## The prediction works in our notebook

In our first machine learning project, we trained a model to identify a penguin from two measurements. But how would someone use that model without opening our notebook?

Suppose we give them a form where they can enter the measurements and press Predict. Then they expect the species to appear in that form, without having to run our Python cells themselves.

So we need to connect the trained model to an application that can use it. That process of preparing and releasing the model for use is called deployment. And when the application is doing its intended job with actual inputs, we're using the model in production.

Before we connect those parts, let's separate what we've already done from what the application needs to do. During training, we gave logistic regression the measurements and the known species. Then it used those examples to learn its parameters, which are the values that determine its predictions.

For a prediction, we give the trained model the measurements of the bird we want to identify. Then it uses the parameters it already learned and returns a species. That's called inference: using the trained model to make a prediction.

Now, when someone presses Predict again, do we need to train logistic regression again? Take a second and decide.

We can use the same trained model for the next prediction, because its learned parameters are already available. Here the input changes, while the parameters stay fixed. So we need a way to preserve those learned values after we close the notebook.

## Preserve the fitted pipeline

Actually, in our project, we need to preserve more than the logistic regression parameters. We also used a scaler to prepare the measurements before classification. And we joined those steps in a pipeline, so the scaler runs first and the classifier runs second.

That scaler learned the mean, or average value, for each feature from the training birds. And it learned each feature's standard deviation, which measures how spread out those values are. During prediction, it uses those saved statistics to standardize the new measurements.

So we save that fitted pipeline in a file, ready for another program to load and use. You'll hear that saved object called a model artifact. Here the artifact contains both the fitted scaler and the trained classifier.

Alongside it, we record what the pipeline expects: bill length first, bill depth second, and both measured in millimeters. Your application needs that information when it prepares a new input. The saved model and its input requirements have to agree.

And we record the Python packages and versions used by the project, because the prediction program needs a compatible environment. These packages are its dependencies. So when you move your artifact to another computer, you also need the software that can load and run it.

When our prediction program starts, it loads the fitted pipeline into memory. Then each new input can use that loaded pipeline, without reading the file again for every prediction. That's the startup behavior we're choosing for this application.

Here's where beginners can go wrong: they save the classifier and leave out the fitted scaler. But the classifier was trained on standardized measurements, so sending raw measurements changes what its inputs mean. We prevent that mismatch by keeping the fitted preprocessing and classifier together.

So far, we've preserved the trained pipeline and loaded it into a program that can use it. Now we need to get measurements from someone's form into that program. How do those two programs communicate?

## The app requests a prediction

Let's start with the form the person sees. That part of the application is the frontend: it collects input and displays the result. Here the frontend reads the measurements when the person presses Predict.

Then the frontend sends those measurements to the program running our fitted pipeline. That program is part of the backend, which does the processing needed to produce the result. So our backend receives the input, checks it and calls the pipeline's prediction method.

To send that input, the frontend needs to know how to request a prediction from the backend. So we define an API, short for application programming interface. Here we're using a web API: we specify where to send the request, what input to include and what response to expect.

For our prediction request, we use a method called post and a destination called slash predict. You'll hear that destination called an API endpoint, because the backend handles requests there for a particular operation. Here that operation is asking our fitted pipeline for a prediction.

Now the frontend sends this as an HTTP request. HTTP is the communication protocol that lets the frontend send a web request and receive a response. So our request has a method, a destination and some data for the backend to process.

Let's use the bird from our first project, with a bill length of forty nine point eight millimeters and a bill depth of sixteen point eight millimeters. We attach each value to its feature name and write those fields as JSON. That's a text format that programs can use to exchange named values.

Those JSON fields become the request body, which is the input data we're sending with the request. Then the frontend sends that body to the prediction endpoint. Now the backend has the measurements, but it still needs to check and arrange them before calling the model.

Here our form gets its features directly from the person using it. A recommendation application might prepare features from user interactions and product records instead. Either way, your application has to select and prepare the information its model expects.

## The backend checks and predicts

Back in our penguin application, let's check the input we received. We need both measurements as valid numbers, so we reject missing values and values such as infinity. That's input validation: checking whether the input meets our requirements.

Suppose bill depth is missing from the request. Then our backend returns an input error, and the frontend asks the person to provide that measurement. That request finishes without running the classifier, because we do not yet have a valid input row.

But even when a value is numeric, we need to know what it measures. Someone could enter four point nine eight centimeters into a field that expects millimeters. That number would pass a numeric type check, but the unit would be wrong for our model.

If we choose to accept centimeters, the application needs to convert them explicitly. Four point nine eight times ten gives forty nine point eight millimeters. Then the prepared input agrees with the unit used during training.

Now we have the right fields and units, and our pipeline expects length first and depth second. What if the backend puts depth first and length second? Take a second and decide.

The pipeline would interpret the depth value as length and the length value as depth. So we've given it different inputs, even though both values are numbers. We avoid that by reading the fields by name and building the row in the order our fitted pipeline expects.

Now we can pass that ordered row to the loaded pipeline. The scaler standardizes both measurements using its saved training values, and logistic regression classifies the transformed row. For this familiar bird, the model's prediction is Gentoo.

Next, the backend puts that predicted species into its response, together with an identifier for the model version it used. If we investigate a result later, we can check which model produced it. Then the response travels back to the frontend.

And the frontend reads the response and displays Gentoo in the result field. That's how the model's output becomes something the person can use. But it's still a predicted species, so a successful response does not tell us whether the prediction is correct.

So far, the frontend has collected measurements, the backend has checked and ordered them, and the fitted pipeline has made a prediction. Then the response gave the frontend a result to display. We've connected the whole request, but this is only one way an application can use a model.

## Three ways production applications use predictions

In our form, inference happens when someone requests a prediction, and the application waits for the result. That's called online inference. And a recommendation service can use the same pattern when an application requests recommendations during someone's visit.

For that recommendation example, the model returns a ranking of products, and the application decides which results to display. So the prediction and the application behavior are separate steps. When you build an application, you need to decide what it should do with the prediction.

But suppose a team wants a periodic report of customers who may stop using its service. That's customer churn prediction, and the project must define what counts as stopping and over what period. Here the team needs a collection of results to review, so we can arrange inference around that report instead of a person's click.

For that periodic report, do we need a live request from a website for every customer? Or could a scheduled program process the records together? Take a second and decide.

We can have a scheduled program read the new records, prepare their features and run predictions for the group. Then it writes the results to a file or database for the team to review. That's batch inference: processing a collection of inputs together.

When the team opens that report, the predictions have already been computed. Then someone can review the results and decide whether to contact a customer. Again, the model estimates the defined outcome, while the organization decides what action is appropriate.

So online and batch inference help us arrange when predictions are produced. A person waiting for an answer and a team reading a scheduled report have different needs. We choose a suitable approach by asking when the application needs the result and how current its input needs to be.

Now we've decided when to produce predictions, but where should inference run? A phone application could load its model on the phone and process an input there. That's on device inference, because the model runs on the device using it.

For example, a phone could use a local model to recognize text in an image. It processes the image on the phone, then the application displays the recognized text. So that inference step can work without sending the image to a remote model server, although other app functions may still need a network.

So on device describes where prediction happens, while online and batch describe how we arrange it. A device can make a prediction when requested or process a collection of inputs. When you plan your application, ask both when the result is needed and where the model can run.

## Where the program runs

Let's return to our penguin form and its backend. We chose to have the frontend request a prediction from a separate running program. Where do we put that program so the intended users can reach it?

During development, we might run it on our own computer and connect through localhost. Localhost refers to the machine making the request. So a person entering that address on another computer would be referring to their computer, not ours.

So for our remote service, we need a suitable computer or managed service where the backend can run. Providing that environment is called hosting. Then we put our application package there, provide its dependencies and start the prediction program.

That running program accepts requests, so we call it a server. You'll also hear people call the computer hosting it a server, but we need to distinguish the machine from the program. Because the computer can be running while the prediction program has stopped.

And our frontend needs an address that reaches that running program. The hosting and network configuration determine whether the intended users can connect. So uploading the model file is only part of the work; we also need the application running and receiving requests.

Now suppose that prediction program stops while someone is using the form. Their request can no longer get a prediction from it. When we restart the program, it loads the saved pipeline again and resumes processing inputs.

For an online service, we need the program available when requests arrive. A batch job can start, process its records and stop until the next run. And an on device model runs within the device's application, so the operating requirements depend on the inference pattern we chose.

## A production model still needs checking

So far, we know how input reaches the model, how the output is used and where prediction runs. Once people start using your application, you need to check that it keeps doing its job. And that means checking both the running program and the quality of the model's predictions.

We've already seen a request fail when the prediction program stopped. So we record request failures and look for changes that need investigation. And we keep expected input validation errors separate from failures inside the service, because they need different fixes.

And even successful requests can take too long for the application using them. The time from sending a request to receiving its response is latency. So you measure that time to find out whether your service is responding quickly enough for its application.

But a service can respond successfully and still predict the wrong species. To check prediction quality, we need reliable information about what the species actually was. Then we can compare the model's prediction with that label, as we did in our first project.

But in a production application, those labels may arrive later, or require someone to collect them. For a churn model, we have to wait for the defined outcome period to learn whether a customer stopped using the service. So returning predictions quickly and evaluating them accurately are different activities.

While we're waiting for labels, we can also check the data reaching the model. Suppose the measurements arriving now are much larger than the measurements used during training. That change in the input distribution, meaning the pattern of values we receive, is called input drift.

For our project, that change could come from different birds or a change in how measurements are recorded. But the changed input distribution does not, by itself, tell us how often predictions are wrong. So we examine the data and compare predictions with reliable labels when those labels are available.

If that investigation shows a model update is needed, we collect suitable training data and train a candidate separately. Then we evaluate its predictions on suitable held-out data and check that it works with the application. Meanwhile, ordinary prediction requests continue to use the released model while we do that work.

When a candidate meets our release requirements, we can deploy it as a new model version. And we keep track of which version produced each result, while retaining a reviewed earlier version. If the new release causes a problem, restoring the earlier version is called rollback.

Now we can follow the wider workflow: collect suitable data, train and evaluate a model, preserve the fitted pipeline and release it into the application. Then observe the inputs, the service and the predictions, and evaluate an update when it's needed. And training remains separate from the inference happening in your application throughout that workflow.

And inside that workflow, our penguin prediction is still the same computation. The application supplies the expected measurements, the fitted pipeline predicts a species, and the frontend displays it. So deployment gives that computation a place in an application someone can use and that we can keep checking.

Try this with a model you'd like to use: decide when the application needs its predictions and where inference should run. Draw the input, preprocessing, model and action, then name one service check and one prediction quality check. You can start with our penguin project, so you already have a trained model to explain.
