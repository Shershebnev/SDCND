**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./images/undistortion_example.png "Undistorted"
[image2]: ./images/undistortion_test_example.png "Road Transformed"
[image3]: ./images/different_threshold_combinations.png "Threshold Combinations"
[image4]: ./images/thresholded_image.png "Thresholded Image Example"
[image5]: ./images/warped.png "Warp Example"
[image6]: ./images/sliding_window.png "Fit Visual"
[image7]: ./images/pipeline_result.png "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf. 

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the `get_calibration_matrix` function in the `image_preparation.py:9`. 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I've experimented with various combinations of gradients and HLS thresholding, here are some examples:
![alt text][image3]

For HLS I've used S (saturation) channel as it seems to be working quite well in different lightning conditions and with different colors of lane lines. For gradients, however, absolute gradient values, magnitude and direction work well to some extent. I've ended up combining using bitwise-OR the thresholding of absolute values of gradients in X direction and S-channel thresholding, because this combination clearly detects the lines without producing too much extra noise and using more computational power.  
Here's an example of my output for this step applied to the example image shown above:

![alt text][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is in the `main` function in the `main.py:19`. Since the camera position was fixed I've decided to hardcode the source and destination points such that the source points create a trapezoid over the approximate lane region. Destination points were chosen so that the converted lines are parallel.

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 580, 460      | 200, 0        | 
| 710, 460      | 1080, 0       |
| 1150, 720     | 1080, 720     |
| 220, 720      | 200, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

For identifying lane-line pixels I've used sliding window approach. Briefly, the initial X coordinate positions for sliding windows for left and right lane are identified using histogram calculated from binarized and warped image. All nonzero pixels at these coordinates plus some margin are considered to be from the lane if the region contains at least `minpix` number of pixels. Then the window is recentered based on those found pixels and is slid up the image. Those steps are repeated until the end of image is reached. Since each consecutive frame in the video won't be really different from the previous one, finding lanes in consecutive frame can be simplified by using the coordinates, found on the previous frame.  
Visualization of the method:
![alt text][image6]

The corresponding functions `find_lanes_initial` and `find_lanes_w_previous` can be found in the `lane_localisation.py:5` and `lane_localisation.py:80` respectively.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Curvature is calculated using `get_curvature` function from `lane_localisation.py:113` and position is calculated using `get_center_offset` function from `lane_localisation.py:131`.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step at the end of `pipeline` function in `pipeline.py:`.  Here is an example of my result on a test image:

![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./test_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The gradient/color thresholds and/or their combination can be improved, as the current solution doesn't seem to work if the lanes are too dull (i.e. in the challenge video) or too bright. Fitting method for lanes can also be improved, since the 2nd degree polynomial doesn't do a great job when there are several turns in quick succession, creating S-shaped turns. This probably also could be fought using smaller region for perspective transform.
