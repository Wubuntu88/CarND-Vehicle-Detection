import cv2
import numpy as np
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import windowing.windower as w
import filtering.filter as f

svc = joblib.load('../../saved_models/svc_rgb_all.pkl')
x_scaler = joblib.load('../../saved_models/x_scaler_rgb_all.pkl')
file_name = 'test1.jpg'
bgr_image = cv2.imread(filename='../../test_images/' + file_name)
rgb_image = cv2.cvtColor(src=bgr_image, code=cv2.COLOR_BGR2RGB)

small_windows = w.slide_window(img=rgb_image, xy_window=[64, 64], y_start_stop=[400, 464], xy_overlap=[0.5, 0.5])
medium_windows = w.slide_window(img=rgb_image, xy_window=[96, 96], y_start_stop=[400, 592], xy_overlap=[0.75, 0.75])
large_windows = w.slide_window(img=rgb_image, xy_window=[128, 128], y_start_stop=[400, 680], xy_overlap=[0.75, 0.75])

triggered_windows_small = w.search_windows(img=rgb_image, windows=small_windows, classifier=svc, scaler=x_scaler)
triggered_windows_medium = w.search_windows(img=rgb_image, windows=medium_windows, classifier=svc, scaler=x_scaler)
triggered_windows_large = w.search_windows(img=rgb_image, windows=large_windows, classifier=svc, scaler=x_scaler)

all_windows = []
all_windows.extend(triggered_windows_small)
all_windows.extend(triggered_windows_medium)
all_windows.extend(triggered_windows_large)

heat_map = np.zeros_like(rgb_image[:, :, 0]).astype(np.float)
f.add_heat(heatmap=heat_map, bbox_list=all_windows)

# window_img = w.draw_boxes(img=rgb_image, bboxes=triggered_windows_small, color=(0, 255, 0), thick=6)
# window_img = w.draw_boxes(img=window_img, bboxes=triggered_windows_medium, color=(0, 64, 255), thick=6)
# window_img = w.draw_boxes(img=window_img, bboxes=triggered_windows_large, color=(255, 255, 0), thick=6)
#
plt.title("Heat map for " + file_name, fontsize=18)
plt.imshow(heat_map, cmap='hot')
plt.show()
