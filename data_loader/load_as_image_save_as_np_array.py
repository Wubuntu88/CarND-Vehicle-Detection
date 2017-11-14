import glob
import cv2
import numpy as np

vehicle_file_names = glob.glob(pathname='../image_data/vehicles/**/*.png', recursive=True)
non_vehicle_file_names = glob.glob(pathname='../image_data/non-vehicles/**/*.png', recursive=True)

vehicle_images = []
for f_name in vehicle_file_names:
    bgr_image = cv2.imread(filename=f_name)
    rgb_image = cv2.cvtColor(src=bgr_image, code=cv2.COLOR_BGR2RGB)
    vehicle_images.append(rgb_image)
non_vehicle_images = []
for f_name in non_vehicle_file_names:
    bgr_image = cv2.imread(filename=f_name)
    rgb_image = cv2.cvtColor(src=bgr_image, code=cv2.COLOR_BGR2RGB)
    non_vehicle_images.append(rgb_image)

vehicle_numpy_img_array = np.array(vehicle_images)
non_vehicle_numpy_img_array = np.array(non_vehicle_images)
print('vehicle images shape: ', vehicle_numpy_img_array.shape)
print('non vehicle images shape: ', non_vehicle_numpy_img_array.shape)

np.save(file='../matrix_image_data/vehicle_images_matrix.npy',
        arr=vehicle_numpy_img_array)
np.save(file='../matrix_image_data/non_vehicle_images_matrix.npy',
        arr=non_vehicle_numpy_img_array)

