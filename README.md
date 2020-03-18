# COVID-19 Classifier

This project uses Keras to build a neural network to detect whether a person has COVID-19 by looking at the X-Ray chest images. The dataset contains 25 images of X-Rays for patients with COVID-19 and 25 images of X-Rays for healthy people.

The classifier uses VGG16 as the base model and gives 90-92% accuracy with 100% sensitivity and 80% specificity. True positives are identified 100% of the time and true negatives are identified 80% of the time.

You can find the tutorial on how to do this on [PyImageSearch](https://www.pyimagesearch.com/2020/03/16/detecting-covid-19-in-x-ray-images-with-keras-tensorflow-and-deep-learning/).

You can get the dataset from this github or look at the tutorial to download it and find out how it was built.