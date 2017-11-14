import numpy as np
import cv2
from typing import List, Tuple
import filtering.history as h
from scipy.ndimage.measurements import label


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        try:
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        except ValueError:
            print('labels sum: ', np.sum(labels[0]))
            print('car numbers: ', labels[1])
            break
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img


def compute_centroids_from_labels(labels: Tuple[object, int]) -> List[Tuple[int, int]]:
    centroid_list = []
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        min_x = int(np.min(nonzerox))
        min_y = int(np.min(nonzeroy))
        max_x = int(np.max(nonzerox))
        max_y = int(np.max(nonzeroy))

        centriod = (max_x - min_x, max_y - min_y)
        centroid_list.append(centriod)
    return centroid_list


def filter_windows(rgb_image, windows, history: h.History) -> Tuple[object, int]:
    heat_map = np.zeros_like(rgb_image[:, :, 0]).astype(np.float)
    add_heat(heatmap=heat_map, bbox_list=windows)

    history.add_or_replace_heat_map(heat_map=heat_map)
    avg_heat_map = history.averaged_heatmap()

    heat_map = apply_threshold(heatmap=avg_heat_map, threshold=2)

    labels = label(heat_map)

    if labels[1] == 0:
        return np.zeros_like(labels[0]), 0

    centroids = compute_centroids_from_labels(labels=labels)

    car_count = labels[1]
    for car_number in range(1, labels[1]+1):
        cent_idx = car_number - 1
        cent = centroids[cent_idx]
        match = history.matches_history(new_centroid=cent)
        if not match:
            labels_matrix = labels[0]
            labels_matrix[labels_matrix == car_number] = 0
            # labels[0][labels[0] == car_number] = 0
            car_count -= 1

    history.add_or_replace_centroids(centroids=centroids)
    return labels[0], car_count

