import feature_extraction.feature_extractor as fe
import data_loader.training_data_loader as tdl
import feature_extraction.hog_extractor as he

import matplotlib.pyplot as plt

vehicles_array, non_vehicles_array = \
    tdl.load_training_data(car_images_file_path='../../matrix_image_data/vehicle_images_matrix.npy',
                           non_car_images_file_path='../../matrix_image_data/non_vehicle_images_matrix.npy')

img = vehicles_array[0]
big_data = he.get_hog_features(multi_channel_img=img)
print(big_data.shape)

features, hog_image = he.get_single_channel_hog_features_and_images(img=img[:, :, 0])
print(features.shape)

plt.imshow(img)
plt.show()
plt.imshow(hog_image, cmap='gray')
plt.show()
