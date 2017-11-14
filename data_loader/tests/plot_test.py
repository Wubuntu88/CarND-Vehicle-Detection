import matplotlib.pyplot as plt
import data_loader.training_data_loader as tdl

vehicles_mtx, non_vehicles_mtx = \
    tdl.load_training_data(car_images_file_path='../../matrix_image_data/vehicle_images_matrix.npy',
                           non_car_images_file_path='../../matrix_image_data/non_vehicle_images_matrix.npy')
a = vehicles_mtx[0, :]
print(a.shape)

plt.imshow(a)
plt.show()
