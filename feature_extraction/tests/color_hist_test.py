import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import feature_extraction.color_histogram as ch

bgr_image = cv2.imread(filename='../../image_data/vehicles/GTI_MiddleClose/image0000.png')
rgb_image = cv2.cvtColor(src=bgr_image, code=cv2.COLOR_BGR2RGB)

# hist_1, hist_2, hist_3, bincen, feature_vec = ch.color_hist_rgb(rgb_image, nbins=32, bins_range=(0, 256))
# color_titles_letters = ['R', 'G', 'B']
# hist_1, hist_2, hist_3, bincen, feature_vec = ch.color_hist_hls(rgb_image, nbins=32, bins_range=(0, 256))
# color_titles_letters = ['H', 'L', 'S']
hist_1, hist_2, hist_3, bincen, feature_vec = ch.color_hist_hsv(rgb_image, nbins=32, bins_range=(0, 256))
color_titles_letters = ['H', 'S', 'V']

# Plot a figure with all three bar charts
if hist_1 is not None:
    fig = plt.figure(figsize=(12, 3))
    plt.subplot(131)
    plt.bar(bincen, hist_1[0])
    plt.xlim(0, 256)
    plt.title(color_titles_letters[0] + ' Histogram')
    plt.subplot(132)
    plt.bar(bincen, hist_2[0])
    plt.xlim(0, 256)
    plt.title(color_titles_letters[1] + ' Histogram')
    plt.subplot(133)
    plt.bar(bincen, hist_3[0])
    plt.xlim(0, 256)
    plt.title(color_titles_letters[2] + ' Histogram')
    fig.tight_layout()
    plt.show()
else:
    print('Your function is returning None for at least one variable...')