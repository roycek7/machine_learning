"""
@author: roycek
"""

import math

import matplotlib.pyplot as plt
import numpy as np


def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def calculate_centroids(data, cluster_centroid, K):
    new_centroid, sse = [], 0
    for _ in range(K):
        new_centroids = [data[i] for i in range(len(data)) if cluster_centroid[i] == _]
        mean_centroid = np.mean(new_centroids, axis=0)
        new_centroid.append(mean_centroid)
        for i in range(len(new_centroids)):
            sse += np.square(new_centroids[i] - new_centroid[_])
    print(f'SSE: {sse}')
    return new_centroid


def euclidean_distance(test_data_vector, training_data_vector):
    """calculates the euclidean distance between two vectors"""
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(test_data_vector, training_data_vector)]))


def assign_clusters(data, centers):
    clusters_list = []
    for i in range(len(data)):
        eu_distance = []
        for center in centers:
            eu_distance.append(euclidean_distance(center, data[i]))
        cluster = -1
        for index, value in enumerate(eu_distance):
            if value == min(eu_distance):
                cluster = index
        clusters_list.append(cluster)
    return clusters_list


def get_convergence(old_c, current_centroid, K):
    total_sum = 0
    for _ in range(K):
        total_sum += np.linalg.norm(old_c[_] - current_centroid[_])
    return total_sum


clustering_data = np.genfromtxt('simple.txt')
clustering_data = np.delete(clustering_data, np.s_[0], axis=0)
# clustering_data = np.delete(clustering_data, np.s_[0], axis=1)

K = 3
clustering_data = normalize_data(clustering_data)
centroid = [clustering_data[np.random.randint(0, len(clustering_data))] for i in range(K)]
initial_clusters = assign_clusters(clustering_data, centroid)

plt.scatter(clustering_data[:, 0], clustering_data[:, 1])
plt.scatter([centroid[i][0] for i in range(len(centroid))], [centroid[i][1] for i in range(len(centroid))],
            marker='*', color=["g"], s=300)

new_clusters, old_centroid = [], []
n_c = False
for i in range(20):
    centroid = calculate_centroids(clustering_data, new_clusters[i - 1] if n_c else initial_clusters, K)
    new_clusters.append(assign_clusters(clustering_data, centroid)), old_centroid.append(centroid)
    if n_c:
        if get_convergence(old_centroid[i - 1], centroid, K) == 0:
            break
    n_c = True


plt.scatter([centroid[i][0] for i in range(len(centroid))], [centroid[i][1] for i in range(len(centroid))],
            marker='o', color=["r"], s=200)
plt.show()
