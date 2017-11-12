import numpy as np
import cv2
from typing import TypeVar

import feature_extraction.color_histogram as ch
import feature_extraction.color_spatial as cs
import feature_extraction.hog_extractor as he


def extract_features(images_array: TypeVar("np.array")) -> list:
    features = []
    for index in range(0, images_array.shape[0]):
        rgb_image = images_array[index]  # this is a (64, 64, 3) numpy array
        spatial_features = cs.bin_spatial(img=rgb_image, size=(32, 32))
        # print('spatial saved_features shape:', spatial_features.shape)
        rgb_hist_features = ch.color_hist_rgb(rgb_img=rgb_image, features_only=True)
        # print('rgb histogram saved_features shape:', rgb_hist_features.shape)
        rgb_hog_features = he.get_hog_features(multi_channel_img=rgb_image)
        # print('rgb hog saved_features shape:', rgb_hog_features.shape)
        img_features = np.concatenate((spatial_features, rgb_hist_features, rgb_hog_features))
        features.append(img_features)
    return features


