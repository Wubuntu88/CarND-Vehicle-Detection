import numpy as np


def load_training_data(car_images_file_path, non_car_images_file_path):
    vehicle_images_matrix = np.load(file=car_images_file_path)
    non_vehicle_images_matrix = np.load(file=non_car_images_file_path)
    return vehicle_images_matrix, non_vehicle_images_matrix

