## Image Processing

This is a collection of algorithms for image processing like real time classificaiton or motion detection.
The real time classification works with a research model from tensorflow and has 90 classes.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for demo purposes.

First you have to install some dependencies.

```
pip2 install tensorflow-gpu
pip2 install tensorflow==1.14
pip2 install --user numpy
pip2 install --user imutils
pip2 install --user opencv-python
pip2 install --user Cython
pip2 install --user contextlib2
pip2 install --user pillow
pip2 install --user lxml
pip2 install --user matplotlib
```

### Motion Detection
For motion detection via background subtraction use the following line and pay attention that the first captured frame of the webcam is only the background. In the frame is an indication if the room is occupied of motion.

![motion detection](https://www.pyimagesearch.com/wp-content/uploads/2015/05/frame_delta_example.jpg)

--video: optional video file, if not provided the webcam is used
--min-area: optional minimum area that gets detected as movement

```
python2 motion_detection.py
```

### Real Time Image Classification
For real time image classification with a live position monetoring use this one.
The used model is a [research model](https://github.com/tensorflow/models/tree/master/research/object_detection) from tensorflow and has 90 classes, but it is light enough to run on mobile devices.

![classification](https://techcrunch.com/wp-content/uploads/2017/06/image1.jpg)
```
python2 real_time_classification.py
```

## Author

* **Julian Streibel** [julianstreibel](https://github.com/julianstreibl)

