# Wildlife Photo Identification Model

Colorado Parks and Wildlife performed rehabilitation on wet meadows in the Gunnison Basin, starting in 2011. Wildlife motion-activated cameras were deployed in 4 treatment sites and 2 control sites. The aim of the study was to determine the success of these rehabilitated sites.

CPW manually counted the instances of different animals in all 316,635 photos. This project, partnered with Boulder AI, aims to create a model that can detect, classify, and record the counts of different animals in the CPW study. The goal is to create a product that will benefit the CPW study, and any other studies like it - based on Boulder AI's unique hardware and software offerings.

BoulderAI provided the image sets from the CPW study, training images, and associated annotations. 

#### NEED: A resource saving model for the CPW Gunnison Basin project.

## The Project

This project serves two purposes: replicate the CPW Gunnison Basin Project with ML algorithims and show Boulder AI's potential value add.

This project's success will be scored on two goals:
1) Accurately replicate the CPW human counted results
2) The project's scalability and robustness

## Models

This models utilizes a Region of Interest Pooling Convolutional Neural Net. There are 3 main object detection / classification algorithms currently en vogue. In increasing accuracy they are:
 - YOLO
 - SSD
 - Faster R-CNN

Although YOLO is considerably quicker than Faster R-CNN, it is less accurate. Interestingly, I had a hard time finding an actual comparison between YOLO and Faster R-CNN. Luckily, math: Faster R-CNN is 10x faster than Fast R-CNN (1), YOLO is 100X faster than Fast R-CNN (2). 100 / 10 = 10; so YOLO is 10 x faster than Faster R-CNN. YOLO performs poorly on small objects, SSD performs adequately on small objects, and Faster R-CNN performs the best.

![Comapre Object Detection Networks](http://cv-tricks.com/wp-content/uploads/2017/12/Size-wise-comparison-of-various-detectors.png)
(3)

The data contains images of small animals, (e.g. birds). I will start with Faster R-CNN and an SSD algorithm.

## Data

The test data consists of photos (some color, some B&W) with labels in a few different places (a few .csv's, and a few .accb's).
The training data consists of pictures, with bounding boxes and labels. I will have to pipe the training photos through a TFR Record File formatter for it to work with the TensorFlow API. 

Here is a test image:



## The model

I am using the TensorFlow API Object detection framework (https://github.com/tensorflow/models/tree/master/research/object_detection). A fair amount of pipelining is required to get the data streamed through to the API in the format it needs. I built my pipeline out so any images could go into a training folder and any images could go into the true test set (read: not validation). This will make it easier for future projects to be quickly built out.

The crux of this project seems to be the tight time constraint, I ended up having a week with the training data. My model quickly learned where to draw bounding boxes. Here is a zoomed out view of the models architecture:


R-CNN's have two moduels: a deep, fully connected network for region proposal and a Fast R-CNN detector, that uses the preposed regions. The first module, also called an RPN (region proposal network) outputs rectangular proposals and an objectness score. It works by sliding a network (nxn window == k) over a convolutional map after every convolution layer. This network is then fed into a lower dimensional feature and then into 2 sibling layers (reg and cls).

The reg layer propososes 4 * k outputs, the bounding box coordinates. The cls layer puts out 2 * k outputs, the objectness of the proposed boxes. 

With more time I could get the model to also classify the animals it detects.

## Results

The model is good at detecting animals (a precision of 83% on a test set of the CPW photos). Here is the same image from the data section with it's predicted bounding box:

## Repo guide

There are 2 main directories, the tensorflow_things directory that makes a model and a web_app directory that runs the webapp. You'll need to put the tensorflow/models directory into each of these directories if you want to run this straight without changing anything. There is a better descrition 

## Sources

1) http://cv-tricks.com/object-detection/faster-r-cnn-yolo-ssd/
2) https://pjreddie.com/darknet/yolo/
3) http://cv-tricks.com/wp-content/uploads/2017/12/Size-wise-comparison-of-various-detectors.png
