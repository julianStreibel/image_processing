## Image Processing

This is a collection of algorithms for image processing like real time classificaiton or motion detection.
The real time classification works with a research model from tensorflow and has 90 classes.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for demo purposes.

First you have to install some dependencies.

```
pip2 install tensorflow-gpu
pip2 install tensorflow
pip2 install --user numpy
pip2 install --user imutils
pip2 install --user opencv-python
pip2 install --user Cython
pip2 install --user contextlib2
pip2 install --user pillow
pip2 install --user lxml
pip2 install --user matplotlib
```

For motion detection via background subtraction use the following line and pay attention that the first captured frame of the webcam is only the background.

--video: optional video file, if not provided the webcam is used
--min-area: optional minimum area that gets detected as movement

```
python2 motion_detection.py
```

For real time image classification with a live position monetoring use this one:

```
python2 real_time_classification.py
```

## Author

* **Julian Streibel** [julianstreibel](https://github.com/julianstreibl)
