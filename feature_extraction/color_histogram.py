import numpy as np
import cv2


# Define a function to compute color histogram features
def color_hist_rgb(rgb_img, nbins=32, bins_range=(0, 256), features_only=True):
    # Compute the histogram of the RGB channels separately
    rhist = np.histogram(rgb_img[:, :, 0], bins=nbins, range=bins_range)
    ghist = np.histogram(rgb_img[:, :, 1], bins=nbins, range=bins_range)
    bhist = np.histogram(rgb_img[:, :, 2], bins=nbins, range=bins_range)
    # Generating bin centers
    bin_edges = rhist[1]
    bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges) - 1]) / 2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((rhist[0], ghist[0], bhist[0]))
    # Return the individual histograms, bin_centers and feature vector
    if features_only:
        return hist_features
    else:
        return rhist, ghist, bhist, bin_centers, hist_features


# Define a function to compute color histogram features
def color_hist_hls(rgb_img, nbins=32, bins_range=(0, 256), features_only=True):
    # Compute the histogram of the RGB channels separately
    HLS_image = cv2.cvtColor(src=rgb_img, code=cv2.COLOR_RGB2HLS)
    h_hist = np.histogram(HLS_image[:, :, 0], bins=nbins, range=bins_range)
    l_hist = np.histogram(HLS_image[:, :, 1], bins=nbins, range=bins_range)
    s_hist = np.histogram(HLS_image[:, :, 2], bins=nbins, range=bins_range)
    # Generating bin centers
    bin_edges = h_hist[1]
    bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges) - 1]) / 2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((h_hist[0], l_hist[0], s_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    if features_only:
        return hist_features
    else:
        return h_hist, l_hist, s_hist, bin_centers, hist_features


def color_hist_hsv(rgb_img, nbins=32, bins_range=(0, 256), features_only=True):
    # Compute the histogram of the RGB channels separately
    HSV_image = cv2.cvtColor(src=rgb_img, code=cv2.COLOR_RGB2HSV)
    h_hist = np.histogram(HSV_image[:, :, 0], bins=nbins, range=bins_range)
    s_hist = np.histogram(HSV_image[:, :, 1], bins=nbins, range=bins_range)
    v_hist = np.histogram(HSV_image[:, :, 2], bins=nbins, range=bins_range)
    # Generating bin centers
    bin_edges = h_hist[1]
    bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges) - 1]) / 2
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((h_hist[0], s_hist[0], v_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    if features_only:
        return hist_features
    else:
        return h_hist, s_hist, v_hist, bin_centers, hist_features
