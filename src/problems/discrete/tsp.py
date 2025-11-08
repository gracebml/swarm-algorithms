import numpy as np
from typing import List, Tuple, Optional
import csv
import os

class TSPProblem:
    def __init__(self, problem_file: str = None):
        if problem_file is not None:
            if not os.path.exists(problem_file):
                raise FileNotFoundError(f"Problem file not found ({problem_file})")
            self._load_from_csv(problem_file)
        else:
            raise ValueError("A problem file must be provided")

        self.eta = np.zeros((self.dim, self.dim))
        for i in range(self.dim):
            for j in range(self.dim):
                if i != j and self.distance_matrix[i, j] > 0:
                    self.eta[i, j] = 1.0 / self.distance_matrix[i, j]

    def _load_from_csv(self, filename: str):
        coordinates = []
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    try:
                        x, y = float(row[0]), float(row[1])
                        coordinates.append([x, y])
                    except ValueError:
                        continue

        if len(coordinates) == 0:
            raise ValueError(f"Invalid file structure ({filename})")

        self.coordinates = np.array(coordinates)
        self.dim = len(coordinates)
        self.name = f"TSP-{self.dim}"
        self.distance_matrix = self._compute_distance_matrix(self.coordinates)

    def _compute_distance_matrix(self, coordinates: np.ndarray):
        n = coordinates.shape[0]
        dist_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(coordinates[i] - coordinates[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

        return dist_matrix

    def get_dim(self):
        return self.dim

    def _nearest_neighbor_tour_length(self):
        n = self.dim
        visited = [False] * n
        tour = []

        current_city = np.random.randint(0, n)
        tour.append(current_city)
        visited[current_city] = True

        total_distance = 0.0

        while len(tour) < n:
            min_dist = float('inf')
            next_city = -1

            for city in range(n):
                if not visited[city] and self.distance_matrix[current_city, city] < min_dist:
                    min_dist = self.distance_matrix[current_city, city]
                    next_city = city

            if next_city == -1:
                break

            total_distance += min_dist
            current_city = next_city
            tour.append(current_city)
            visited[current_city] = True

        total_distance += self.distance_matrix[tour[-1], tour[0]]

        return total_distance