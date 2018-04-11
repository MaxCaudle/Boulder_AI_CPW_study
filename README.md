# Wildlife Photo Identification Model

Colorado Parks and Wildlife performed rehabilitation on wet meadows in the Gunnison Basin, starting in 2011. Wildlife motion-activated cameras were deployed in 4 treatment sites and 2 control sites. The aim of the study was to determine the success of these rehabilitated sites.

CPW manually counted the instances of different animals in all 316,635 photos. This project, partnered with Boulder AI, aims to create a model that can detect, classify, and record the counts of different animals in the CPW study. The goal is to create a product that will benefit the CPW study, and any other studies like it - based on Boulder AI's unique hardware and software offerings.

BoulderAI provided the image sets from the CPW study, training images, and associated annotations. 

#### NEED: A resource saving model for the CPW Gunnison Basin project.

## The Project

This project serves two purposes: replicate the CPW Gunnison Basin Project with ML algorithims and show Boulder AI's potential value add.

This project's success will be scored on two goals:
1) Accurately replicate the CPW human counted results
2) The project's expandability and robustness

## Models to consider

This will utilize a Convolutional Neural Net. There are 3 main object detection / classification algorithms currently en vogue. In increasing accuracy they are:
 - YOLO
 - SSD
 - Faster R-CNN

Although YOLO is considerably quicker than Faster R-CNN, it is less accurate. Interestingly, I had a hard time finding an actual comparison between YOLO and Faster R-CNN. Luckily, math: Faster R-CNN is 10x faster than Fast R-CNN (1), YOLO is 100X faster than Fast R-CNN (2). 100 / 10 = 10; so YOLO is 10 x faster than Faster R-CNN. YOLO performs poorly on small objects, SSD performs adequately on small objects, and Faster R-CNN performs the best.

![](http://cv-tricks.com/wp-content/uploads/2017/12/Size-wise-comparison-of-various-detectors.png)
(3)

The data contains images of small animals, (e.g. birds). I will start with Faster R-CNN and an SSD algorithm.

## Data

The test data consists of photos (some color, some B&W) with labels in a few different places (a few .csv's, and a few .accb's).
The training data consists of pictures, with bounding boxes and labels. I will have to pipe the training photos through a TFR Record File formatter for it to work with the TensorFlow API. 

## The model

I am using the TensorFlow API Object detection framework (https://github.com/tensorflow/models/tree/master/research/object_detection). A fair amount of pipelining is required to get the data streamed through to the API in the format it needs. I built my pipeline out so any images could go into a training folder and any images could go into the true test set (read: not validation). This will make it easier for future projects to be quickly built out.

The crux of this project seems to be the tight time constraint, I ended up having a week with the training data. Originally, the training data looked like it would come in one form and I configured a pipeline for that. Once I got the data though, I had to drastically reconfigure my pipelines so they could handle the different formats.

My model quickly learned where to draw bounding boxes, but after a day of training it couldn't categorize the animals correctly. I changed some of the hyper parameters and switched models. I let it train for a few days and the model still couldn't categorize the animals very well. With a few days left before the model was due, I decided to reduce the number of classes to the 3 most common animals: cows, deer, and horses.

After training for a few days the model still can't get the animals categorization correct. I think this is because the animals are similair in apperance, and a shortage of training data. 

## Results

The model is good at detecting animals (a precision of 83% on a test set of the CPW photos). 

## Repo guide

There are 2 main directories, the tensorflow_things directory that makes a model and a web_app directory that runs the webapp. You'll need to put the tensorflow/models directory into each of these directories if you want to run this straight without changing anything. There is a better descrition 

## Sources

1) http://cv-tricks.com/object-detection/faster-r-cnn-yolo-ssd/
2) https://pjreddie.com/darknet/yolo/
3) http://cv-tricks.com/wp-content/uploads/2017/12/Size-wise-comparison-of-various-detectors.png
