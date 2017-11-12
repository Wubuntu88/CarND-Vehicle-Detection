import numpy as np
import time

before = time.time()

images_matrix = np.load(file='../matrix_image_data/non_vehicle_images_matrix.npy')

print(images_matrix.shape)

after = time.time()

print('diff:', after - before)