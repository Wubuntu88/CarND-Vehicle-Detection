import feature_extraction.feature_extractor as fe
import data_loader.training_data_loader as tdl
import feature_extraction.hog_extractor as he

import matplotlib.pyplot as plt
import cv2

bgr_img = cv2.imread(filename='../../test_images/test1.jpg')
rgb_img = cv2.cvtColor(src=bgr_img, code=cv2.COLOR_BGR2RGB)

# junk1, hog_img_1 = he.get_single_channel_hog_features_and_images(img=rgb_img[:, :, 0], vis=True)
# junk2, hog_img_2 = he.get_single_channel_hog_features_and_images(img=rgb_img[:, :, 1], vis=True)
junk3, hog_img_3 = he.get_single_channel_hog_features_and_images(img=rgb_img[:, :, 2], vis=True)

plt.title("test1.jpg blue channel hog visualization", fontsize=20)
plt.imshow(hog_img_3, cmap='gray')
plt.show()
