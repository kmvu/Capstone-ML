# Machine Learning Engineer Nanodegree
## Capstone Project
Khang Vu
December 31st, 2050

## I. Definition ##

### Project Overview ###

#### Background ####
Consider the case when we are passionate about flowers, and we are curious at the same time about what type of flower they are, what name people usually call them, etc.

What if we can do just that with a little help from technology? Imagine we can use our own phone to take pictures of the flowers, and right away, its name appears afterwards on the screen, and we satisfy. Here comes a use case where we can apply Machine Learning (ML) algorithm to make predictions.

This classification use case is one of the problem hosted by `Kaggle`, where:

> Kaggle is a platform for predictive modelling and analytics competitions in which statisticians and data miners compete to produce the best models for predicting and describing the datasets uploaded by companies and users. This crowdsourcing approach relies on the fact that there are countless strategies that can be applied to any predictive modelling task and it is impossible to know beforehand which technique or analyst will be most effective.[Kaggle]

[Click here](https://www.kaggle.com/c/oxford-102-flower-pytorch) for more information in Kaggle, about `Oxford 102 Flower Pytorch - 102 Flower Classification Created by Enthusiast's` competition. Even though the competition requires a solution in Pytorch, we will instead use Keras in this project.

***References***

- [Kaggle](https://medium.com/neuralspace/kaggle-1-winning-approach-for-image-classification-challenge-9c1188157a86 "https://medium.com/neuralspace/kaggle-1-winning-approach-for-image-classification-challenge-9c1188157a86")

#### Purposes and Motivation ####
The main goal for this project is to create an intelligent Machine Learning (ML) model using different techniques to help us recognize most of the common flower types. And this model could be used to integrate into mobile apps, devices to help predict and open more opportunities for developers to innovate, especially if they are passionate about flowers.

This project will be very helpful and diverse in technical term, by using various ML techniques from Supervised Learning, Exploratory Data Analysis (EDA), to Deep Learning, etc. Apart from that, recognizing objects has been an interesting topic in recent years since it can make our applications smarter by learning and making predictions by themselves in different categories without being explicitly programmed, which is interestingly motivated to put in the efforts.

#### Datasets and Inputs ####

There are 102 flower categories commonly occurring in the United Kingdom. [*Maria-Elena Nilsback*][1] and [*Andrew Zisserman*][2], in *Department of Engineering Science* at the **University of Oxford**, have decided to create a dataset, corresponding to the aforementioned 102 flower categories, or so-called **classes** interchangeably. In details, each class consists of **40 to 258 images**. Visualization about each class (name, image, label number, etc.) can be found at [this site][3].

According to [Visual Geometry Group][4] at the University of Oxford:

> The images have large scale, pose and light variations. In addition, there are categories that have large variations within the category and several very similar categories. The dataset is visualized using isomap with shape and colour features.(4)

***References***:
- http://www.robots.ox.ac.uk/~men/
- http://www.robots.ox.ac.uk/~az/
- http://www.robots.ox.ac.uk/~vgg/data/flowers/102/categories.html
- http://www.robots.ox.ac.uk/~vgg/data/flowers/102/

[1]: http://www.robots.ox.ac.uk/~men/ "http://www.robots.ox.ac.uk/~men/"
[2]: http://www.robots.ox.ac.uk/~az/ "http://www.robots.ox.ac.uk/~az/"
[3]: http://www.robots.ox.ac.uk/~vgg/data/flowers/102/categories.html "http://www.robots.ox.ac.uk/~vgg/data/flowers/102/categories.html"
[4]: http://www.robots.ox.ac.uk/~vgg/data/flowers/102/ "http://www.robots.ox.ac.uk/~vgg/data/flowers/102/"


### Problem Statement ###

#### Quantifiable ####
Given a batch of different types of flowers, and we want to classify them by matching with their corresponding type names. In other words, we will label these flower types by printing their corresponding names under their images as results. In order to figure out the corresponding names for the flowers, we can calculate the probabilities for each **classes** represented by an output layer from a Deep Neural Network, which should produces the maximum likelihood of those classified names.

> A deep neural network (DNN) is an artificial neural network (ANN) with multiple layers between the input and output layers. The DNN finds the correct mathematical manipulation to turn the input into the output, whether it be a linear relationship or a non-linear relationship. The network moves through the layers calculating the probability of each output.(5)

And from there, we can predict the corresponding name for the image by taking the class with largest probability within 102 output predictions. These outputs are represented in form of probabilities because we use [Softmax function][5] to calculate probabilities distribution across our **classes**.

> In mathematics, the softmax function, also known as **softargmax** or **normalized exponential function**, is a function that takes as input a vector of K real numbers, and normalizes it into a probability distribution consisting of K probabilities.

#### Measurable ####
*Accuracy* is a metric we can use to measure our predicting performance since we can clearly observe the percentage of how many images (each represents one flower type) being correctly classified out of 102 flower types.

#### Replicable ####
This classification problem should be reproducible by taking images of different flowers and making predictions accordingly again and again.

#### Solution Statement ####

Firstly, we need to make sure our dataset is clean by pre-processing it using various Exploratory Data Analysis ([EDA][6]) concept. Then we will be using a Deep Neural Networks (DNN) at the core to train our data, and calculate the probabilities as final output after making predictions, which will potentially tell us the flower types. We will also use Image Augmentation technique to vary the input types so that the network can learn better in terms of diversity, as part of data pre-processing step. We then use `Accuracy`, and `F1-score` metrics to see how accurate our model performs after training.

> Exploratory Data Analysis refers to the critical process of performing initial investigations on data so as to discover patterns, to spot anomalies, to test hypothesis and to check assumptions with the help of summary statistics and graphical representations.(6)

After all, developers can make use of [CoreML (iOS)][7] or [ML Kit (Android)][8] to convert and incorporate this ML model into their platforms and start making predictions right on their respective devices as real applications.

***References***:
- [Softmax function - wikipedia](https://en.wikipedia.org/wiki/Softmax_function)
- [Deep Neural Network - wikipedia](https://en.wikipedia.org/wiki/Deep_learning)
- https://towardsdatascience.com/exploratory-data-analysis-8fc1cb20fd15
- https://developer.apple.com/machine-learning/
- https://developers.google.com/ml-kit/

[5]: https://en.wikipedia.org/wiki/Softmax_function "https://en.wikipedia.org/wiki/Softmax_function"
[6]: https://towardsdatascience.com/exploratory-data-analysis-8fc1cb20fd15 "EDA"
[7]: https://developer.apple.com/machine-learning/ "https://developer.apple.com/machine-learning/"
[8]: https://developers.google.com/ml-kit/ "https://developers.google.com/ml-kit/"

### Metrics ###
In this section, you will need to clearly define the metrics or calculations you will use to measure performance of a model or result in your project. These calculations and metrics should be justified based on the characteristics of the problem and problem domain. Questions to ask yourself when writing this section:
- _Are the metrics you’ve chosen to measure the performance of your models clearly discussed and defined?_
- _Have you provided reasonable justification for the metrics chosen based on the problem and solution?_

As mentioned in Problem Statement, we will be using `Accuracy` to measure our model's performance. In some cases, `Accuracy` is not enough to measure our performance properly and we need to use F-beta score by determining whether we need a high `Recall` score or a high `Precision` score. However, in this case, either `False Positives` or `False Negatives` is not a big problem, because they just simply mean "No, this is not a *xyz* type". So we can use `Accuracy` to measure our model performance.

*Accuracy* can be calculated as following:
```
Accuracy = (True Positives + True Negatives) / 102 (dataset size)
```

where:
  * **True Positives** is the number of images which are correctly classified / predicted.
  * **True Negatives**, which is not available in this case for a classification problem since we only have either "this type of flower" or "that type of flower".

Hence the formula can be shortened as:

```
Accuracy = True Positives / 102
```

Optionally, we can multiply the result by 100 to turn it into percentage format, if necessary.

## II. Analysis ##

### Data Exploration ###

The dataset contains good amount of different flower types (102), with a number of images represent each **class**. Hence, we can directly use them to start training our model without having to bring in more data images from another sources to fill up missing **classes**.

Within this dataset, we have three folders *Training*, *Validation*, and *Testing*, which stand for their own purposes, respectively. Each folder should contains 102 categories, and each category contains **40 to 258 images**. *Training* and *Validation* folders are used for training our model, then we use *Testing* folder to validate our model after training in order to avoid ***overfitting problem***. This can count as part of the pre-processing step for our dataset. Additionally, It is better to verify that our model performs well with new set of images (and not just from images that it already knew). This way we can raise our confidence that it can predict flowers' names that it has never seen before.

*Validation* set contains **818 images**, while *Training* set contains **6652 images**. We recognize that *Validation* set takes about **11%** total number of images original for training by following calculation:

```
[818 / (818 + 6652)] * 100 = 10.95%
```

We need a *Validation* set in order to verify the accuracy of our model after training so that the *Testing* set will be used to test our model's performance without worrying about the case where our model learns too deep into the testing dataset and unable to predict accurately new unseen data (**Overfitting**).

Since our dataset is mainly based on images, not statistical numbers, we don't have to worry about outliers or missing values messing up our data.

### Exploratory Visualization ###

The following screenshot shows five different flower types, with five images per each category (per row):

![Visualization sample dataset](/images/visualization.png)

Our model will extract the basic characteristics from the images and try to learn patterns from its features, then combine these knowledges to learn even more common complex patterns between the same type of flower and predict the likelihood of flower types eventually as a goal.

### Algorithms and Techniques ###

In order to solve this flower classification problem, a **Deep Learning architecture** incorporated with **Transfer Learning** technique will be used to learn patterns, extract feature characteristics from images and then combine these knowledges to come up with the most likelihood probabilities for the predicted flowers. We, hence, use this result to classify our flower categories.

Before developing our own **Deep Learning** architecture, we need to make sure our problem is solvable and can be improved, fine-tuned later. A simple **benchmark model** will be used in this case as a starting point, which will be disussed in the next section.

After comparing with the aforementioned benchmark model, we now can use **Transfer Learning** technique to benefit the pre-trained model (VGG-19) by using its *pre-trained weights* to continue training and fit our own problem.

#### Parameters ####

A pre-trained model like VGG-19 will need mandatory parameters such as:

 * ***weights***: *'imagenet'* - is chosen in this case, since we want to reuse the pre-trained weights through 1000 images in **imageNet** database.

 * ***include_top***: *'False'* - is set so we can ignore the output trained from imageNet and use our own custom output layers.

 * ***input_shape***: this parameter specifies an expected size to feed in VGG-19 network (224 width x 224 height x 3 color channels).

 * ***loss***: *'categorical_crossentropy'* - since our problem is multiclass classification, this parameter is proved to produce more accurate result then *'binary-crossentropy'* option.

 * ***optimizers***: *'Adam'*, and ***Learning rate***: *'0.00005''* - this optimizer and learning rate combination has been experimented to reach the optimal result better, even though it takes quite some times to converge to a local minimum (with help from GPU power).

#### Strategy (Algorithm) ####

Since our dataset is considered as a small dataset (< 10,000 images), and quite similar to **imageNet** database, we need a strategy to make sure we train the pre-trained model in an optimized way to fit our problem. For this scenario, we will `freeze` all the pre-trained layers in the original network. In other words, we don't train the original layers again, but using the pre-trained weights from **imageNet** instead.

Doing this will allow us to add new extra custom layers in place of the original output layer, so that the model can produce the desire result uniquely to our flower problem.

After adding new custom layers, we are then ready to train this new architecture network. And since we already froze all the original layers from the pre-trained network, when we train this new entire network again, the original layers' weights will not get updated in the process, but only in the extra layers. This way we can purposely train our extra layers to make sure they gain enough basic knowledges about this classification problem.

After a preset number of `epochs` of training, and this new model is supposed to learn some new knowledges, we will un-freeze a few layers in the original network and let the entire network be trained again for another amount of epochs. This time we should be able to get a model with decent enough accuracy to make predictions.

***Note:***

* `epochs`: number of times we do a **forward pass** and a **backward propagation** to update the weights in each layers.

* `forward pass`: this procedure happens when we feed an image into out DNN from the first layer and **forward** pass it through each layers until the last layer in the network to get the features extracted along the way.

* `backward propagation`: this procedure happens when the errors have been calculated from the last layer by using a *loss function* (`Cross Entropy` in this case) and passed **backward** to update the weights in each layers.

### Benchmark ###

It is a good idea to setup a benchmark model, which plays a role of making sure that our classification problem is solvable and giving us an idea on how this problem can be solved with just a simple model architecture. Later on, we can use this result to have a confidence that a DNN using Transfer Learning will definitely produce a better result.

This simple benchmark model using in this scenario only includes one Convolutional layer (2D), and one Dense layer as core layers. Then we add a Pooling layer to help reduce space info to filter out features, and a Dropout layer to give each layer a chance to train in the network. Lastly, we need a Dense layer as an output layer to make prediction using a `Softmax` activation function to make sure the results are in form of probabilities. The architecture is as following:

```
bench_model = Sequential()
bench_model.add(Conv2D(256, kernel_size=2,
  input_shape=(224, 224, 3), activation='relu'))
bench_model.add(GlobalAveragePooling2D())
bench_model.add(Dropout(0.5))
bench_model.add(Dense(128, activation='relu'))
bench_model.add(Dropout(0.2))
bench_model.add(Dense(train_generator.num_classes,
  activation='softmax'))
```
![Benchmark model](/images/benchmark_model.png)

After five epochs, this benchmark model gives an accuracy of `12.20%`, which is not really but expected, since this model knows nothing about our current dataset.

We will need a model with more knowledge on lower levels, to extract the foundation features out from the given images, using a pre-trained model such as VGG-19, which has already been trained on ImageNet through 1000 images. This way we can modify this pre-trained model using Transfer Learning technique to solve our unique problem with customized Dense layers.

## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing
In this section, all of your preprocessing steps will need to be clearly documented, if any were necessary. From the previous section, any of the abnormalities or characteristics that you identified about the dataset will be addressed and corrected here. Questions to ask yourself when writing this section:
- _If the algorithms chosen require preprocessing steps like feature selection or feature transformations, have they been properly documented?_
- _Based on the **Data Exploration** section, if there were abnormalities or characteristics that needed to be addressed, have they been properly corrected?_
- _If no preprocessing is needed, has it been made clear why?_

### Implementation

The `include_top` parameter when we initialize the VGG-19 network  makes sure we don't include the original output layer. Hence, in order to fit to our problem, we need to create our custom layers to replace the top layer (output layer) from the original network. We will add a few `Dense` layers and apply some `Dropout` functions to avoid **Overfitting**. It will look like following:

```
## Freezing all the layers from the original network
for layer in base_model.layers:
    layer.trainable = False

# Add some custom layers
net = Flatten(name='flatten')(base_model.output)
net = Dense(4096, activation='relu')(net)
net = Dropout(0.5)(net)
net = Dense(4096, activation='relu')(net)
net = Dropout(0.5)(net)
net = Dense(train_generator.num_classes,
            activation='softmax')(net) # 102 output classes
```

This way, once we start training this new customized network, we only spend time to actually train the newly added custom layers while making use of the existing weights for the original layers, which gives our custom layers chances to learn  our dataset to some extend. I mentioned `to some extend` because we don't entirely train this new architecture for too long (just about **10 epochs** and we stop).

At this point, after `10 epochs` of experimentation, the model has been noticed to produce an accuracy of `65%`, not the best performance but it does show that the learning progress has good potential to learn even more and better if we keep training since both *Training* and *Validation* accuracies are simultaneously increasing with a consistent distant gap in between, as following:

![Training progress](/images/training_progress.png)

Before proceeding, we save the weights that have been trained so far as a checkpoint for later use.

Next, we will give our model another chance to learn more. But this time, we will un-freeze the last five layers of the original network (from the Convolutional layers) and let them be trained along with our custom layers. This strategy is known to be quite efficient to train network with a small dataset. What we will do is to load the weights from out checkpoint earlier and start training again using the loaded weights for about `20 epochs`. This process has been experimented and we get an accuracy of `91%`, which is not a bad performance after all for a classification problem!

The training/validation progress has been recorded as below:

![final_training_progress](/images/final_training_progress.png)


In this section, the process for which metrics, algorithms, and techniques that you implemented for the given data will need to be clearly documented. It should be abundantly clear how the implementation was carried out, and discussion should be made regarding any complications that occurred during this process. Questions to ask yourself when writing this section:
- _Is it made clear how the algorithms and techniques were implemented with the given datasets or input data?_
- _Were there any complications with the original metrics or techniques that required changing prior to acquiring a solution?_
- _Was there any part of the coding process (e.g., writing complicated functions) that should be documented?_

### Refinement
In this section, you will need to discuss the process of improvement you made upon the algorithms and techniques you used in your implementation. For example, adjusting parameters for certain models to acquire improved solutions would fall under the refinement category. Your initial and final solutions should be reported, as well as any significant intermediate results as necessary. Questions to ask yourself when writing this section:
- _Has an initial solution been found and clearly reported?_
- _Is the process of improvement clearly documented, such as what techniques were used?_
- _Are intermediate and final solutions clearly reported as the process is improved?_


## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
In this section, the final model and any supporting qualities should be evaluated in detail. It should be clear how the final model was derived and why this model was chosen. In addition, some type of analysis should be used to validate the robustness of this model and its solution, such as manipulating the input data or environment to see how the model’s solution is affected (this is called sensitivity analysis). Questions to ask yourself when writing this section:
- _Is the final model reasonable and aligning with solution expectations? Are the final parameters of the model appropriate?_
- _Has the final model been tested with various inputs to evaluate whether the model generalizes well to unseen data?_
- _Is the model robust enough for the problem? Do small perturbations (changes) in training data or the input space greatly affect the results?_
- _Can results found from the model be trusted?_

### Justification
In this section, your model’s final solution and its results should be compared to the benchmark you established earlier in the project using some type of statistical analysis. You should also justify whether these results and the solution are significant enough to have solved the problem posed in the project. Questions to ask yourself when writing this section:
- _Are the final results found stronger than the benchmark result reported earlier?_
- _Have you thoroughly analyzed and discussed the final solution?_
- _Is the final solution significant enough to have solved the problem?_


## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
In this section, you will need to provide some form of visualization that emphasizes an important quality about the project. It is much more free-form, but should reasonably support a significant result or characteristic about the problem that you want to discuss. Questions to ask yourself when writing this section:
- _Have you visualized a relevant or important quality about the problem, dataset, input data, or results?_
- _Is the visualization thoroughly analyzed and discussed?_
- _If a plot is provided, are the axes, title, and datum clearly defined?_

### Reflection
In this section, you will summarize the entire end-to-end problem solution and discuss one or two particular aspects of the project you found interesting or difficult. You are expected to reflect on the project as a whole to show that you have a firm understanding of the entire process employed in your work. Questions to ask yourself when writing this section:
- _Have you thoroughly summarized the entire process you used for this project?_
- _Were there any interesting aspects of the project?_
- _Were there any difficult aspects of the project?_
- _Does the final model and solution fit your expectations for the problem, and should it be used in a general setting to solve these types of problems?_

### Improvement
In this section, you will need to provide discussion as to how one aspect of the implementation you designed could be improved. As an example, consider ways your implementation can be made more general, and what would need to be modified. You do not need to make this improvement, but the potential solutions resulting from these changes are considered and compared/contrasted to your current solution. Questions to ask yourself when writing this section:
- _Are there further improvements that could be made on the algorithms or techniques you used in this project?_
- _Were there algorithms or techniques you researched that you did not know how to implement, but would consider using if you knew how?_
- _If you used your final solution as the new benchmark, do you think an even better solution exists?_

-----------

**Before submitting, ask yourself. . .**

- Does the project report you’ve written follow a well-organized structure similar to that of the project template?
- Is each section (particularly **Analysis** and **Methodology**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your analysis, methods, and results?
- Have you properly proof-read your project report to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
- Is the code that implements your solution easily readable and properly commented?
- Does the code execute without error and produce results similar to those reported?
