# Wildlife Photo Identification Model

Colorado Parks and Wildlife performed rehabiliation on wet meadows in the Gunnison Basin, starting in 2011. Wildife motion-activated cameras were deployed in 4 treatment sites and 2 control sites. The aim of the study was to determine the success of these rehabilited sites.

CPW manually counted the instaces of different animals in all 316,635 photos. This project, partnered with Boulder AI, aims to create a model that can detect, classify, and record the counts of different animals in the CPW study. The goal is to create a product that will benefit the CPW study, and any other studies like it - based on Boulder AI's unique hardware and software offerings. 

#### NEED: A resource saving model for the CPW Gunnison Basin project.

## The Project

This project serves two purposes: replicate the CPW Gunnison Basin Project with ML algorithims and show Boulder AI's potential value add.

This project's success will be scored on three goals:
1) Accurately replicate the CPW human counted results
2) The project's expandability and robustness
3) Demonstrated additional value add Boulder AI's hardware will make possible

## Models to consider

This will utilize a Convolutional Neural Net. There are 3 main object detection / classification algorithms currently en vogue. In increasing accuracy they are:
 - YOLO
 - SSD
 - Faster R-CNN
 
Although YOLO is considerably faster than Faster R-CNN, it is less accurate. Interstingly, I had a hard time finding an actual comparison between YOLO and Faster R-CNN. Luckily, math: Faster R-CNN is 10x faster than Fast R-CNN (1), YOLO is 100X faster than Fast R-CNN (2). 100 / 10 = 10; so YOLO is 10 x faster than Faster R-CNN. YOLO performances poorly on small objects, SSD performs adequately on small objects, and Faster R-CNN performs the best.

![](http://cv-tricks.com/wp-content/uploads/2017/12/Size-wise-comparison-of-various-detectors.png)
(3)

The data contains images of small animals, (e.g. birds). I will start with Faster R-CNN.

## Data

The test data consists of photos (some color, some B&W) with labels in a few different places (a .csv, some .accb's, and ).
The training data consists of pictures, with bounding boxes and labels. I will have to pipe the training photos through a TFR Record File formatter for it to work with the TensorFlow API


## Sources

1) http://cv-tricks.com/object-detection/faster-r-cnn-yolo-ssd/
2) https://pjreddie.com/darknet/yolo/
3) http://cv-tricks.com/wp-content/uploads/2017/12/Size-wise-comparison-of-various-detectors.png
