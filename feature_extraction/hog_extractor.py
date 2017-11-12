from skimage.feature import hog
import numpy as np


def get_hog_features(multi_channel_img, num_orientations=9, pix_per_cell=8, cell_per_block=2):
    # Call with two outputs if vis==True
    hog_channel_list = []
    for channel_index in range(0, multi_channel_img.shape[2]):
        single_channel = multi_channel_img[:, :, channel_index]
        features = hog(single_channel, orientations=num_orientations, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       visualise=False, feature_vector=True)
        hog_channel_list.append(features)
    to_return = np.ravel(hog_channel_list)
    return to_return


def get_single_channel_hog_features_and_images(img, num_orientations=9, pix_per_cell=8, cell_per_block=2):
    features, hog_image = hog(img, orientations=num_orientations, pixels_per_cell=(pix_per_cell, pix_per_cell),
                              cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                              visualise=True, feature_vector=True)
    return features, hog_image

