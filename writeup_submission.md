##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

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
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

[rgb_img_test1]: ./test_images/test1.jpg
[img_hog_feat_red_ch_test1]: ./output_images/hog_images/red_channel_hog_test1.png
[img_hog_feat_green_ch_test1]: ./output_images/hog_images/green_channel_hog_test1.png
[img_hog_feat_blue_ch_test1]: ./output_images/hog_images/blue_channel_hog_test1.png

[static_detect_test1]: ./output_images/static_car_detections/test1.png
[static_detect_test3]: ./output_images/static_car_detections/test3.png
[static_detect_test5]: ./output_images/static_car_detections/test5.png
[static_detect_test6]: ./output_images/static_car_detections/test6.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  

This document is the writeup submission for project 5.

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for extracting the hog features is located in the feature_extraction/hog_extractor.py.

Here is the code snippet:
```python
from skimage.feature import hog
import numpy as np
def get_hog_features(multi_channel_img, num_orientations=9,
                     pix_per_cell=8, cell_per_block=2, is_feature_vector=True):
    """
    This method takes a multichannel image as input.
     It loops through the each channel, and extracts the hog features for that channel.
     It then assembles the hog features as a feature vector by the ravel method.
    :return: np.array of features
    """
    hog_channel_list = []
    for channel_index in range(0, multi_channel_img.shape[2]):
        single_channel = multi_channel_img[:, :, channel_index]
        features = hog(single_channel, orientations=num_orientations, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       visualise=False, feature_vector=is_feature_vector)
        hog_channel_list.append(features)
    to_return = np.ravel(hog_channel_list)
    return to_return
```
This method is used in the extract_features.py file in the feature_extraction module.  
The feature extraction module is used in the search_windows method in the windower.py file inside the windowing module.

I chose to extract hog features from the RGB color channels.
I thought these would be satisfactory, so I chose to go with the RGB channels and explore other color spaces if I needed to.

By visualizing the hog features of an image, we can see that the RGB channels do give us outlines of the cars.

|Original test1.jpg | Hog features Red Channels test1.jpg |
|:-------------------------:|:-------------------------:|
|![rgb_img_test1] | ![img_hog_feat_red_ch_test1]|

|Hog features Green Channels | Hog features Blue Channels |
|:-------------------------:|:-------------------------:|
|![img_hog_feat_green_ch_test1] | ![img_hog_feat_blue_ch_test1]|

####2. Explain how you settled on your final choice of HOG parameters.

I chose the HOG parameters that were in the Udacity course. The parameters all are legitimate.

Here are my parameters:
* num_orientations = 9
* pix_per_cell = 8
* cell_per_block = 2

As for the number of orientations, I believe 9 yields enough granularity for the histogram of the gradients.
Creating more orientations would most likely achieve diminishing returns.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the default parameters in sklearn (C = 1.0).
I trained on both the GIT and KTTI image data.
This yielded an accuracy of around 99%.
Because the accuracy was so good, I ended up keeping the first classifier I trained.

I used the following features to train my classifier:
* Raw colors in the pixels as a 1-d feature vector (in feature_extraction/color_spatial.py)
* RGB color histograms (in feature_extraction/color_histogram.py - I used the rgb method).
* HOG features (in feature_extraction/hog_extractor.py).

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used the first set of code from Udacity to implement my sliding window search.
It is located in the windowing/windower.py file.
I used the slide_window() to find all of my windows, then the search windows and decide if there is a car in that window.

I decided on several window sizes: (64x64), (96x96), and (128x128).
For the above window sizes, I chose the following configurations:
###### (64x64)
- I chose y_start=400 and ystop=464.  
This means that there was only one row of small windows and that was just below the horizon where one would imagine small cars to be.
I restricted the small windows to this area because the small windows were triggering too many false positives when allowed to creep to far down into the frame.
- I chose and overlap of 0.5.  I chose this small overlap because a high overlap made the video generation take too long.
- Occasionally, I would omit these windows in my video generation because they made my code run too long.

###### (96x96)
- I chose y_start=400 and y_stop=592.
I wanted it to start at the horizon to capture cars at the horizon, even if they are smaller than the bounding box.
I chose to stop at 592 because that seemed like a reasonable limit where a  medium scaled car would be.
-I chose a 0.75 overlap because with a larger size window, I thought I could afford the extra overlap, and I wanted as many windows as was reasonable.

###### (128x128)
- I chose y_start=400 and y_stop=650.
I chose it to start at the horizon to capture as many cars as possible.
I chose it to stop at 650 so that it would stop just shy of the hood of the car.
-I chose a 0.75 overlap.  Like in the (96x96) example, I wanted as many windows as was reasonable.
With the large window size, I though I could afford the high overlap.


####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Here are some images of my pipeline locating cars in an image: (Note, I use filtering techniques that use several previous frames for )

##### Car detection without filtering

| test1.jpg | test3.jpg |
|:-------------------------:|:-------------------------:|
|![static_detect_test1] | ![static_detect_test]|

|test5.jpg | test6.jpg |
|:-------------------------:|:-------------------------:|
|![static_detect_test5] | ![static_detect_test6]|

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.
From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.
I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.
I then assumed each blob corresponded to a vehicle.
I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

