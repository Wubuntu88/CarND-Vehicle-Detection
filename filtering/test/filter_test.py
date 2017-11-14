import cv2
import numpy as np
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import windowing.windower as w
import filtering.filter as f
import filtering.history as h
from scipy.ndimage.measurements import label

svc = joblib.load('../../saved_models/svc_rgb_all.pkl')
x_scaler = joblib.load('../../saved_models/x_scaler_rgb_all.pkl')

bgr_image = cv2.imread(filename='../../test_images/test1.jpg')
rgb_image = cv2.cvtColor(src=bgr_image, code=cv2.COLOR_BGR2RGB)

small_windows = w.slide_window(img=rgb_image, xy_window=[64, 64], y_start_stop=[400, 464], xy_overlap=[0.75, 0.75])
medium_windows = w.slide_window(img=rgb_image, xy_window=[96, 96], y_start_stop=[400, 592], xy_overlap=[0.85, 0.85])
large_windows = w.slide_window(img=rgb_image, xy_window=[128, 128], y_start_stop=[400, 680], xy_overlap=[0.85, 0.85])

triggered_windows_small = w.search_windows(img=rgb_image, windows=small_windows, classifier=svc, scaler=x_scaler)
triggered_windows_medium = w.search_windows(img=rgb_image, windows=medium_windows, classifier=svc, scaler=x_scaler)
triggered_windows_large = w.search_windows(img=rgb_image, windows=large_windows, classifier=svc, scaler=x_scaler)

all_windows = []
all_windows.extend(triggered_windows_small)
all_windows.extend(triggered_windows_medium)
all_windows.extend(triggered_windows_large)

history = h.History()

labels = f.filter_windows(rgb_image=rgb_image, windows=all_windows, history=history)

# f.draw_labeled_bboxes(img=rgb_image, labels=labels)
# plt.imshow(rgb_image)
# plt.show()