import cv2
"""
Taken from the Udacity course material
"""


def bin_spatial(img, size=(32, 32)):
    """
    resize and image to a given size and then turns it into a feature vector.
    The image is resized so that the feature vectors are of the same length accross different sized images.
    :param img: An image in any (x, y, depth) dimension.
    :param size: the dimension to resize the image to.
    :return: the feature vector of the image unraveled and turned into a 1d array.
    """
    return cv2.resize(img, size).ravel()
