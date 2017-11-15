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

I trained a linear SVM using...

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

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

