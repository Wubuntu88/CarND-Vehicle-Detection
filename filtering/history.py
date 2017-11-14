from typing import List, Tuple
import math


class History:
    def __init__(self, n_frames: int = 8, match_threshold: int = 3, distance_threshold: float = 150.0):
        self.heat_maps = []
        # self.label_boxes = []
        self.last_n_centroids = []  # list of lists of floats
        self.index_to_replace = 0
        self.n_frames = n_frames
        self.match_threshold = match_threshold
        self.distance_threshold = distance_threshold

    def add_or_replace_heat_map(self, heat_map):
        if len(self.heat_maps) <= self.n_frames:
            self.heat_maps.append(heat_map)
        else:
            self.heat_maps[self.index_to_replace] = heat_map
        self.index_to_replace = (self.index_to_replace + 1) % self.n_frames

    def averaged_heatmap(self):
        if len(self.heat_maps) <= 0:
            return None
        else:
            return sum(self.heat_maps) / len(self.heat_maps)

    # def add_or_replace_label_box(self, label_box):
    #     if len(self.heat_maps) <= self.n_frames:
    #         self.heat_maps.append(label_box)
    #     else:
    #         self.heat_maps[self.index_to_replace] = label_box
    #     self.index_to_replace = (self.index_to_replace + 1) % self.n_frames

    def add_or_replace_centroids(self, centroids: List[Tuple[int, int]]):
        if len(self.last_n_centroids) <= self.n_frames:
            self.last_n_centroids.append(centroids)
        else:
            self.last_n_centroids[self.index_to_replace] = centroids
        self.index_to_replace = (self.index_to_replace + 1) % self.n_frames

    def matches_history(self, new_centroid: Tuple[int, int]) -> bool:
        if len(self.last_n_centroids) < self.match_threshold:
            return False
        time_slices_that_meet_threshold = 0
        for centroids_at_time_slice in self.last_n_centroids:
            for test_centroid in centroids_at_time_slice:
                x_diff_sq = (test_centroid[0] - new_centroid[0]) ** 2
                y_diff_sq = (test_centroid[1] - new_centroid[1]) ** 2
                distance = math.sqrt(x_diff_sq + y_diff_sq)
                if distance < self.distance_threshold:
                    time_slices_that_meet_threshold += 1
                    break
            if time_slices_that_meet_threshold >= self.match_threshold:
                return True
        return False
