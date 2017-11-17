import cv2
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import windowing.windower as w

svc = joblib.load('../../saved_models/svc_rgb_all.pkl')
x_scaler = joblib.load('../../saved_models/x_scaler_rgb_all.pkl')
file_name = 'test5.jpg'
bgr_image = cv2.imread(filename='../../test_images/' + file_name)
rgb_image = cv2.cvtColor(src=bgr_image, code=cv2.COLOR_BGR2RGB)

small_windows = w.slide_window(img=rgb_image, xy_window=[64, 64], y_start_stop=[400, 464], xy_overlap=[0.5, 0.5])
medium_windows = w.slide_window(img=rgb_image, xy_window=[96, 96], y_start_stop=[400, 592], xy_overlap=[0.75, 0.75])
large_windows = w.slide_window(img=rgb_image, xy_window=[128, 128], y_start_stop=[400, 650], xy_overlap=[0.75, 0.75])

triggered_windows_small = w.search_windows(img=rgb_image, windows=small_windows, classifier=svc, scaler=x_scaler)
triggered_windows_medium = w.search_windows(img=rgb_image, windows=medium_windows, classifier=svc, scaler=x_scaler)
triggered_windows_large = w.search_windows(img=rgb_image, windows=large_windows, classifier=svc, scaler=x_scaler)

print(len(triggered_windows_small))
print(len(triggered_windows_medium))
print(len(triggered_windows_large))


window_img = w.draw_boxes(img=rgb_image, bboxes=triggered_windows_small, color=(0, 255, 0), thick=6)
window_img = w.draw_boxes(img=window_img, bboxes=triggered_windows_medium, color=(0, 64, 255), thick=6)
window_img = w.draw_boxes(img=window_img, bboxes=triggered_windows_large, color=(255, 255, 0), thick=6)

plt.title("Large (Yellow) and Medium (Blue) windows triggering (No filtering) - " + file_name, fontsize=20)
# plt.title("Medium (Blue) and Small (Green) windows triggering (No filtering) - " + file_name, fontsize=20)
plt.imshow(window_img)
plt.show()
