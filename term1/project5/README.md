**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/test_examples.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

HOG feature extraction happens in the `train_model.py` in the `main` function and uses some of the functions from `helpers.py`.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

For HOG parameter selection I've employed good old Grid Search with cross-validation :)

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

All the training related staff happens in `train_model.py`. First, features are extracted for all the images in the dataset. Then the dataset is split into training and test sets. Training set is used to first fit a `StandardScaler` from scikit-learn. Both training and test set are then normalized using fitted scaled. After that, Linear Support Vector Classifier (SVC) is trained on training set with the default parameters. Achieved accuracy on the test set is 99%.

For features I've decided to use HOG features, color features and spatial features all together. 

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

First of all, I've limited the search area so that in only covers the bottom part of the images (where the road is). Next I've scanned this part of the image extracting windows of image and running them through classifier with every scan moving to right (or bottom) by half the window size. I've ended up using two scales - smaller window size for the region of the image, that show the further parts of the road and larger window size for the region of the image that is closer to the camera. This made the pipeline run faster (compared to one scale) for obvious reasons of reducing the number of windows.


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. Here are some example images:

![alt text][image3]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./test_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the heatmaps of detections from each subsequent frame of the video up `n_last_heatmaps` frames using the `Heatmap` class.  From them I've created a combined heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline is based on features that are not very robust: there is a huge variety of colors, for example and it is possible that not all of them were present in the dataset and it might no be feasible to cover all the possible colors. Another problem is susceptibility to lightning conditions, same as in the previous project on lane finding. It is also really slow and I'm not sure if it would be possible to up the speed to at least 24 FPS as SVMs are known to be quite bad at scaling both in time and computational resources. 
In another MOOC I've used YOLO neural network architecture for the same task of detecting and drawing bounding boxes around vehicles in the videostream and it has shown to be significantly more robust. It is also possible to run YOLO in real-time (24 FPS) which feels to be crucible for the self-driving car.

