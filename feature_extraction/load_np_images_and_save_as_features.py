import numpy as np

import feature_extraction.feature_extractor as fe
import data_loader.training_data_loader as tdl
import time


vehicles_array, non_vehicles_array = \
    tdl.load_training_data(car_images_file_path='../matrix_image_data/vehicle_images_matrix.npy',
                           non_car_images_file_path='../matrix_image_data/non_vehicle_images_matrix.npy')

before = time.time()

car_features = fe.extract_features(images_array=vehicles_array)
non_car_features = fe.extract_features(images_array=non_vehicles_array)

after = time.time()

print('time: ', (after - before))

np.save(file='../saved_features/rgb_all/car_features.npy', arr=car_features)
np.save(file='../saved_features/rgb_all/non_car_features.npy', arr=non_car_features)

