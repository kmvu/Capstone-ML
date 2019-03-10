# Machine Learning Engineer Nanodegree
## Capstone Proposal
Khang Vu

December 31st, 2050

## Proposal ##
==============

### Domain Background ###

#### Background
Consider the case when we are passionate about flowers, and we are curious at the same time about what type of flower they are, what name people usually call them, etc.

What if we can do just that with a little help from technology? Imagine we can use our own phone to capture the flower, and right away, its name appears afterwards on the screen, and we satisfy. Here comes a use case where we can apply Machine Learning algorithm into the picture.

#### Purposes and Motivation
The main goal for this project is to create an intelligent ML model using different techniques to help us recognize most of the common flower types. And this model could be used to integrate into mobile apps, devices to help predict and open more opportunities for developers to innovate, especially if they are passionate about flowers.

This project will be very helpful and diverse in technical term, by using various Machine Learning (ML) techniques from Supervised Learning, EDA, to Deep Learning, etc. Apart from that, recognizing objects has been an interesting topic in recent years since it can make our applications smarter by learning and making predictions by themselves in different categories without being explicitly programmed, which is interestingly motivated to put in the efforts.

### Datasets and Inputs ###

There are 102 flower categories commonly occurring in the United Kingdom. [*Maria-Elena Nilsback*][1] and [*Andrew Zisserman*][2], in *Department of Engineering Science* at the **University of Oxford**, have decided to create 102 category datasets, corresponding to the aforementioned 102 flower categories, or so-called **classes** interchangeably. In details, each class consists of **40 to 258 images**. Visualization about each class (name, image, label number, etc.) can be found at [this site][3].

According to [Visual Geometry Group][4] at the University of Oxford:

> The images have large scale, pose and light variations. In addition, there are categories that have large variations within the category and several very similar categories. The dataset is visualized using isomap with shape and colour features.

This dataset has already contained good amount of different flower types, and it especially contains images for each **class**. Hence, we can directly use them to start training our model without having to bring in more data images from another sources to fill up missing **classes**. However, since this dataset contains all images in one folder, we need put these images of each flower types into their respective folders so can we load them in and classify them accordingly. We also need to separate them into *Training*, *Testing*, and *Validation* folders for their own purposes. *Training* and *Testing* folders are used for training our model, then we use *Validation* folder to validate our model after training in order to avoid ***overfitting problem***. This can count as part of the pre-processing step for our dataset. Additionally, It is better to verify that our model performs well with new set of images (and not just from images that it already knew). This way we can raise our confidence that it can predict flowers' names that it has never seen before.

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

#### Quantifiable
Given we have a batch of different types of flowers, and we want to classify them by matching with their corresponding type names. In other words, we will label these flower types by printing their corresponding names under their images as results. In order to figure out the corresponding names for the flowers, we can calculate the probabilities for each **classes** represented by output layer from a Deep Neural Network, which should produces the likelihood of those classified names. And from there, we can predict the corresponding name for the image by taking the class with largest probability within 102 output predictions. These outputs are represented in form of probabilities because we use [Softmax function][5] to calculate probabilities distribution across our **classes**.

> In mathematics, the softmax function, also known as **softargmax** or **normalized exponential function**, is a function that takes as input a vector of K real numbers, and normalizes it into a probability distribution consisting of K probabilities.

#### Measurable
*Accuracy* is a metric we can use to measure our predicting performance since we can clearly observe the percentage of how many images (each represents one flower type) being correctly classified out of 102 flower types.

#### Replicable
This classification problem should be reproducible by taking images of different flowers and making predictions accordingly again and again.

***References***:
- [Softmax function - wikipedia](https://en.wikipedia.org/wiki/Softmax_function)

[5]: https://en.wikipedia.org/wiki/Softmax_function "https://en.wikipedia.org/wiki/Softmax_function"

### Solution Statement ###

Firstly, we need to make sure our dataset is clean by pre-processing it using various EDA techniques. Then we will be using a Deep Neural Networks (DNN) at the core to train our data, and calculate the probabilities as final output after making predictions, which will potentially tell us the flower types. We could also use Image Augmentation technique to vary the input types so that the network can learn better in terms of diversity, as part of data pre-processing step. We then use `Accuracy` metric to see how accurate our model performs after training.

After all, developers can make use of [CoreML (iOS)][6] or [ML Kit (Android)][7] to convert and incorporate this ML model into their platforms and start making predictions right on their respective devices as real applications.

***References***:
- https://developer.apple.com/machine-learning/
- https://developers.google.com/ml-kit/

[6]: https://developer.apple.com/machine-learning/ "https://developer.apple.com/machine-learning/"
[7]: https://developers.google.com/ml-kit/ "https://developers.google.com/ml-kit/"

### Benchmark Model ###

Compared to the project [`Dog breed` classification problem][8], as the fourth project in this Machine Learning Nanodegree at Udacity, this flower classification problem can be seen as somewhat similar to the `Dog breed` project. Both have the same task of trying to recognize / predict the objects' type (`dog` and `flower`). We can, indeed, use this `Dog breed` project as our benchmark model to compare our model to, as we will use *Transfer learning* technique and apply to our Deep Neural Network (DNN) to make predictions. Hence, most of the aspects from both projects should be resembled and observed in the same way. (Quantifiable, measurable, replicable)

The model trained in `Dog breed` has reached **71.7703%** of accuracy, and we will try to reach this number or even better for our flower classification problem. Since we as well use `Accuracy` as the main metric to observe the model performance, and also use *Convolutional Neural Network (CNN)*, as one of the powerful type of DNNs out there, to filter spacial information in the images and train our dataset.

**References**

- https://github.com/kmvu/dog_classification

[8]: https://github.com/kmvu/dog_classification "Dog breed - Khang Vu (Github)"

### Evaluation Metrics

As mentioned in Problem Statement, we will be using `Accuracy` to measure our model's performance. In some cases, `Accuracy` is not enough to measure our performance properly and we need use F-beta score to make sure . However, in this case, either `False Positives` or `False Negatives` is not a big problem, because they just simply mean "No, this is not a ... type".

*Accuracy* can be calculated as following:
```
Accuracy = (True Positives + True Negatives) / 102 (dataset size)
```

where:
  * **True Positives** is the number of images which are correctly classified / predicted.
  * **True Negatives**, which is not available in this case for a classification problem since we only have either "This type of flower" or "that type of flower".

Hence the formula can be shortened as:

```
Accuracy = True Positives / 102
```

Optionally, we can multiply the result by 100 to turn it into percentage format, if necessary.

### Project Design ###
_(approx. 1 page)_

####

In this final section, summarize a theoretical workflow for approaching a solution given the problem. Provide thorough discussion for what strategies you may consider employing, what analysis of the data might be required before being used, or which algorithms will be considered for your implementation. The workflow and discussion that you provide should align with the qualities of the previous sections. Additionally, you are encouraged to include small visualizations, pseudocode, or diagrams to aid in describing the project design, but it is not required. The discussion should clearly outline your intended workflow of the capstone project.

---------

## Q&A ##

- Does the proposal you have written follow a well-organized structure similar to that of the project template? **Yes**
- Is each section (particularly **Solution Statement** and **Project Design**) written in a clear, concise and specific fashion? Are there any ambiguous terms or phrases that need clarification?
- Would the intended audience of your project be able to understand your proposal?
- Have you properly proofread your proposal to assure there are minimal grammatical and spelling mistakes?
- Are all the resources used for this project correctly cited and referenced?
