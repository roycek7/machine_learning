"""
@author: roycek
"""

import math

import numpy as np
import pylab
from sklearn.model_selection import train_test_split

label = 1
k = 100
features = 11


def normalize_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def euclidean_distance(test_data_point, training_data_point):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(test_data_point, training_data_point)]))


def accuracy_metric(actual, predicted, correct=0):
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / len(actual) * 100.0


def sort_tuple(tup, y):
    tup.sort(key=lambda x: x[y])
    return tup


def predict_test_data(k_th):
    predicted_labels = []
    for i in range(len(x_test)):
        distance = []
        for j in range(len(x_train)):
            eu_distance = euclidean_distance(x_test[i], x_train[j])
            distance.append((eu_distance, y_train[j]))
        minimum_distant_neighbours = sort_tuple(distance, 0)[:k_th]
        nearest_neighbours = []
        for j in minimum_distant_neighbours:
            nearest_neighbours.append(j[label])
        predicted_labels.append(max(nearest_neighbours, key=nearest_neighbours.count))
    return predicted_labels


numpy_data = np.loadtxt("glass.data", delimiter=",", usecols=range(1, features))
y_label = [i[-label] for i in numpy_data]
scaled_data = normalize_data(numpy_data)
training_data, test_data, y_train, y_test = train_test_split(scaled_data, y_label, train_size=0.8, test_size=0.2,
                                                             shuffle=True)
x_train = training_data[:, :-label]
x_test = test_data[:, :-label]

pylab.plot(range(k - label), [accuracy_metric(y_test, predict_test_data(i)) for i in range(1, k)])
pylab.xlabel('k')
pylab.ylabel('Accuracy')
pylab.show()
