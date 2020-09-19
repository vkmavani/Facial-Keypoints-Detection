# Facial-Keypoints-Detection-

## Project Summary


*  In this project, facial key-points(also called facial landmark) are the small dots shown on each of the faces in the image below. In each training and testing image, there is a single face and 68 key-points with coordinates (x,y) for that face. These key-points are important areas of the face like eye, corners of mouth, nose, etc.

<img src="https://github.com/vkmavani/Facial-Keypoints-Detection/blob/master/Facial%20Keypoints%20Detection/Images/key_pts_example.png">

<I>Key-points with numbers</I>

<img src="https://github.com/vkmavani/Facial-Keypoints-Detection/blob/master/Facial%20Keypoints%20Detection/Images/landmarks_numbered.jpg">

## File-1 : Load and Visualize data

To work with any dataset, first you must have to be familer with dataset. If you are going to work with image data, then you must have to visualize the images and to know about label or target. This dataset is extracted from <a href='https://www.cs.tau.ac.il/~wolf/ytfaces/'>YouTube Faces Data</a>, which includes videos of people in YouTube videos. In this file, first dataset is loaded and display the number of images in data and visualize first image with it's key-points.



## File-2 : Create Model and Training
This dataset consists 5770 color images and key-points with respect to images:
    
   * <I>3462 of these images are training images, to create a model to predict keypoints.</I> 
   * <I>2308 of these images are testing images, which will be used to test the accuracy of the model.</I>

First resize the images with shape (224, 224, 1). The input images of neural network were converted into gray scale so it could be normalized. So, the final shape will be (3462, 224, 224, 1) and Key-points shape is (3462, 68, 2) which is then reshaped as (3462, 136).

Architecture of Model
<img src="https://github.com/vkmavani/Facial-Keypoints-Detection/blob/master/Facial%20Keypoints%20Detection/Images/Model.png">


## File-3 : Facial Keypoints Detection

In this file, key-points is detected using previously trained model.
First all the faces are detected using 'haarcascade_frontalface_default.xml' and co-ordinates from the detection (x,y,w,h) are used to crop the image after then resize to shape (224, 224, 1) as an input to the model to predict the keypoints.

<img src="https://github.com/vkmavani/Facial-Keypoints-Detection/blob/master/Facial%20Keypoints%20Detection/Images/obamas.jpg">
<img src="https://github.com/vkmavani/Facial-Keypoints-Detection/blob/master/Facial%20Keypoints%20Detection/Images/Detection.png">
<img src="https://github.com/vkmavani/Facial-Keypoints-Detection/blob/master/Facial%20Keypoints%20Detection/Images/obamakeypoints.png">
<img src="https://github.com/vkmavani/Facial-Keypoints-Detection/blob/master/Facial%20Keypoints%20Detection/Images/michelle.png">


## File-4 : Webcam - Facial Keypoints Detection
At last by using webcam and previously trained model, live keypoints detection is implemented.
