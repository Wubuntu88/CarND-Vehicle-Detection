## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[sub_clip_0_5_car_classified_yes_confidence]: ./plots/project_video_subclip0-5_yes_predictions_confidence.png
[sub_clip_0_5_car_classified_no_confidence]: ./plots/project_video_subclip0-5_no_predictions_confidence.png
[sub_clip_7_15_car_classified_yes_confidence]: ./plots/project_video_subclip7-15_yes_predictions_confidence_crop.png
[sub_clip_7_15_car_classified_no_confidence]: ./plots/project_video_subclip7-15_no_predictions_confidence_crop.png

[rgb_img_test2]: ./test_images/test2.jpg
[rgb_img_test3]: ./test_images/test3.jpg
[rgb_img_test6]: ./test_images/test6.jpg

[heat_map_img_test1]: ./output_images/heat_maps/test1.png
[heat_map_img_test3]: ./output_images/heat_maps/test3.png
[heat_map_img_test6]: ./output_images/heat_maps/test6.png

[label_on_heat_map_img_test1]: ./output_images/labels_on_heat_maps/test1.png
[label_on_heat_map_img_test3]: ./output_images/labels_on_heat_maps/test3.png
[label_on_heat_map_img_test6]: ./output_images/labels_on_heat_maps/test6.png

[label_on_img_test1]: ./output_images/labels_on_images/test1.png
[label_on_img_test3]: ./output_images/labels_on_images/test3.png
[label_on_img_test6]: ./output_images/labels_on_images/test6.png

[stubborn_false_positive]: ./output_images/_stubborn_false_positive/stubborn_false_positive.png


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  

This document is the writeup submission for project 5.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

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

| Original test1.jpg |
|:-------------------------:|
|![rgb_img_test1] |

| Hog features Red Channels test1.jpg |
|:-------------------------:|
| ![img_hog_feat_red_ch_test1]|

| Hog features Green Channels |
|:-------------------------:|
| ![img_hog_feat_green_ch_test1]|

| Hog features Blue Channels |
|:-------------------------:|
| ![img_hog_feat_blue_ch_test1]|

#### 2. Explain how you settled on your final choice of HOG parameters.

I chose the HOG parameters that were in the Udacity course. The parameters all are legitimate.

Here are my parameters:
* num_orientations = 9
* pix_per_cell = 8
* cell_per_block = 2

As for the number of orientations, I believe 9 yields enough granularity for the histogram of the gradients.
Creating more orientations would most likely achieve diminishing returns.

As for the pixels per cell and cells per block, the values chosen seem to cover enough area to give a good indication of the gradient at that location.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the default parameters in sklearn (C = 1.0).
I trained on both the GIT and KTTI image data.
This yielded an accuracy of around 99%.
Because the accuracy was so good, I ended up keeping the first classifier I trained.

I used the following features to train my classifier:
* Raw colors in the pixels as a 1-d feature vector (in feature_extraction/color_spatial.py)
* RGB color histograms (in feature_extraction/color_histogram.py - I used the rgb method).
* HOG features (in feature_extraction/hog_extractor.py).

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

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


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Here are some images of my pipeline locating cars in an image: (Note, I use filtering techniques that use several previous frames for )

##### Car detection without filtering

| test1.jpg | test3.jpg |
|:-------------------------:|:-------------------------:|
|![static_detect_test1] | ![static_detect_test3]|

|test5.jpg | test6.jpg |
|:-------------------------:|:-------------------------:|
|![static_detect_test5] | ![static_detect_test6]|

To better improve the performance of my classifier, I thresholded the confidence to reduce false positives.
As an investigation, I took two subclips of video time clips 0-5 and 7-15.
In the 0-5 time clip, there is open road without any cars.
My classifier generated many false positives in the center of the road.
In the 7-15 clip, the white car was driving, and my classifier detected it effectively.

For both clips, I ran my classifier and collected two lists for each run:
-The prediction confidence values for each window classified as a vehicle.
-The prediction confidence values for each window classified as a non-vehicle.

Here are graphs for each:

| Subclip 0-5 (all road - false positives) | Subclip 0-5 (all road - 'correct' negative classification) |
|:-------------------------:|:-------------------------:|
|![sub_clip_0_5_car_classified_yes_confidence] | ![sub_clip_0_5_car_classified_no_confidence]|

| Subclip 7-15 (mostly cars - 'correct' positive classification)| Subclip 7-15 (Mostly Cars - 'correct' negative classification) |
|:-------------------------:|:-------------------------:|
|![sub_clip_7_15_car_classified_yes_confidence] | ![sub_clip_7_15_car_classified_no_confidence]|

Observe the two figures on the left.  These figures both show a exponential-like distribution.
Note that for the 0-5 subclip, The maximum prediction confidence is just below 2.5.
That means for this road section, the false positive classifications for the road did not have a prediction confidence above 2.5.

Also observe that the maximum confidence for the 7-15 subclip is 6.5.
In general, the distribution is shifted farther to the right, indicating that the classifier many time is has a higher confidence associated with a car image.

Do note two things:
* The histogram of all road has much fewer samples (Above 100), whereas the histogram of mostly cars has many more samples (many thousand).
* The distributions of confidence in classifications for each case (car and non-car) have significant overlap.
In fact the majority of classifications for either group are below 1.5.
This means that it most likely not be possible to determine whether the positive classification is a road or car based on the confidence.
This does not mean that it does not help.  I list several thresholding values below.

I tried out several threshold levels:
* 1.0: This did a great job of getting rid of the false positives.
It completely removed the false positives from the beginning of the video, where there is a discoloration in the pavement.
However, there was a problem detecting the white car where it went over the bright pavement.
The detection of the white car dropped out for several seconds in that area.
Also, there was a problem detecting the black car as it went towards the horizon.
(That may have been more of an issue because I omited the small windows on the horizon for that run).

* 0.75: This level did a good job of eliminating false positives, but had sections where the white car dropped out on bright pavement.

* 0.65: A good balance between eliminating false positives and not having cars drop out.

* 0.5: This level did a great job of detecting the cars, but had several cases of false positives.
This is especially true in the beginning of the video where the discoloration in the road color triggered the false positives.

* 0.35: This did a good job of detecting the cars, but also had the same false positives as the 0.5 threshold.


---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my submission video](./output_videos/submission_video/project_video_avg_heat_maping_5_frames_with_small_windows_.65_to_.85_overlap_pred_conf_thresh=0.65_no_centroid.mp4)
This video does a good job of rejecting false positives and keeping track of the cars.
There is one instance where it does not get the car outlines correct when the black car moves in front of the white car.

To see another example that handles the black car going in front of the white car better, look at this [link](./output_videos/test/z_pretty_good/project_video_avg_heat_maping_5_frames_with_prediction_confidence_thresh=0.5.mp4).

To see the windows trigger without any filtering, check this out [no filtering](./output_videos/test/test_unfilt_long_1.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

For each Frame, I created a heatmap.
For each window that triggered, the pixels of the heatmap of the corresponding bounding box were incremented by one.

Here are images with their corresponding heat maps: (Note that no averaging takes place in these heat maps - but averaging did occur im my video).

| RGB Image test2.jpg | Heat Map test2.png |
|:-------------------------:|:-------------------------:|
|![rgb_img_test2] | ![heat_map_img_test1]|

| RGB Image test3.jpg | Heat Map test3.png |
|:-------------------------:|:-------------------------:|
|![rgb_img_test3] | ![heat_map_img_test3]|

| RGB Image test6.jpg | Heat Map test6.png |
|:-------------------------:|:-------------------------:|
|![rgb_img_test6] | ![heat_map_img_test6]|

I used this as a way to combine the results of the overlapping classified windows.
To make the detection more robust, I pixelwise summed the heatmaps by 5 frames and then averaged each pixel value by the number of frames.
This entailed storing 5 frames worth of heat maps.  I stored this as a member variable in a custom History object (located in filtering/history.py)
I then applied a threshold of two, meaning for a pixel to be stuck on the heat map, it must be triggered by at least two windows for 5 frames in a row.
I then used the `scipy.ndimage.measurements.label()` method on the averaged heat map to detect the car regions.
This was the most impactful filtering technique I performed.

As one further step of robustness, I also kept track of the centroids for each frame.
For example, the previous frame could have contained 3 centroids.
The current frame could contain two centroids, and so on.
The filtering process is the following: 
- If a candidate label is generated, its centroid is computed.   
- If the centroid is less than 150 pixels for a centroid for a frame for 3 out of the previous 5 frames, 
it is considered legitimate and its labels is allowed to be entered into the heat map.

This extra step helped filter out false positives when I tried it.
However, I believe it caused some side effects where cars dropped out too much,
especially when the black car went towards the horizon at the end of the video.
For this reason, I decided to not include this part in my submission video.

Here are some Images of the labels drawn on the heat maps (without frame averaging):
(Note: the white outline makes the heat maps look less hot)

| RGB Image test1.jpg | Label on Heat Map test1.png |
|:-------------------------:|:-------------------------:|
|![rgb_img_test1] | ![label_on_heat_map_img_test1]|

| RGB Image test3.jpg | Label on Heat Map test3.png |
|:-------------------------:|:-------------------------:|
|![rgb_img_test3] | ![label_on_heat_map_img_test3]|

| RGB Image test6.jpg | Label on Heat Map test6.png |
|:-------------------------:|:-------------------------:|
|![rgb_img_test6] | ![label_on_heat_map_img_test6]|

And here are the images with the labels drawn on them:

| RGB Image test1.jpg | Label on Image test1.png |
|:-------------------------:|:-------------------------:|
|![rgb_img_test1] | ![label_on_img_test1]|

| RGB Image test3.jpg | Label Image test3.png |
|:-------------------------:|:-------------------------:|
|![rgb_img_test3] | ![label_on_img_test3]|

| RGB Image test6.jpg | Label on Image test6.png |
|:-------------------------:|:-------------------------:|
|![rgb_img_test6] | ![label_on_img_test6]|


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?


The issues I faced in this project had mostly to do with filtering.
I had problems of too many false positives.
In particular, I had a stubborn false positive problem where there were vehicle detections in the middle of the road.

![stubborn_false_positive]

To solve this issue, I did several things.  The most effective was to average the heat maps over several frames.
I also tried a technique of keeping heat map sections only if they were close enough to several previous centroids.
However, this did not seem to yield obvious benefits when used in conjunction with the heat map average.
For the submission video, I excluded this part, but I believe it has potential if the distance parameter is tuned.

Another solution to the false positive issue was to use the confidence prediction and use a threshold to classify a new window as an image.
I tried many threshold levels from 0.25 to 1.0.
In the case of 1.0, it did a good job of eliminating false positives, but generated too many false negatives in sequence, so the car label dropped out of too many frames.
For cases of 0.3, 0.5, etc, They captured the cars well, but did not filter out the stubborn false positive.

After much tweaking, I decided on a threshold value of 0.65.
It excelled at removing false positives and false negatives, even though it had a peculiar behaviour when one car went in front of the other.

I also experimented with many window sizes.  I found that an overlap of 0.65 to 0.85 was good.  More windows is probably better, but more windows takes longer.

I can think of several cases where my pipeline may fail:
* When there is discolored pavement, or bright pavement.
* When confronted with shadows over several frames.  This is expecially true if the shadow has pockets in it, so that it has many strong gradients.
* When a car travels in front of another.
* When a car approaches the horizon; the horizon is not always consistent, and my small windows will not always be aligned well enough.

To make the pipeline more robust I could do the following:
* I could explore different classifiers, such as neural networks or decision trees.
* Exploring other color spaces such as HSV, HSL, and LUV could provide better features for my classifier.
* Getting better statistics about the confidence for when a car is classified as a car, and when a when a non car is classified as a car.
This could be done on the leftover test data after the model has been trained.  A more difficult job would be to do it using the video.
This would allow us to better choose a cutoff confidence threshold for deciding if a window should be classified as having a car in it.
* tuning the centroid distance threshold over several frames could prove beneficial.
* using Udacity's training data of cars would bring additional training data for the classifier.

Using these techniques, I can improve my vehicle tracking in the future.
