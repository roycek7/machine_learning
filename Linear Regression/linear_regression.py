"""
@author: roycek
"""

import numpy as np


def normalize_data(data_X):
    return (data_X - data_X.mean()) / data_X.std()


def predict(data_X, weights):
    return np.dot(data_X, weights)


def theta_closed_form(data_X, y_label):
    return np.linalg.pinv(np.dot(data_X.T, data_X)) * np.dot(data_X.T, np.mat(y_label))


def mean_square_error(y_train, y_pred):
    return np.square(np.subtract(y_train, y_pred)).mean() 


def insert_weight_one(train_x):
    weight_ones = np.ones((train_x.shape[0], 1))
    return np.hstack((train_x, weight_ones))


def load_dataset(file):
    dataset = np.loadtxt(file, usecols=range(0, 14))
    x = normalize_data(dataset[:, :-1])
    y = dataset[:, -1].reshape((-1, 1))
    return x, y


train_X, train_y = load_dataset("housing_train.txt")
X = insert_weight_one(train_X)
theta = theta_closed_form(X, train_y)
y_ = predict(X, theta)
mse = mean_square_error(train_y, y_)

test_X, test_y = load_dataset("housing_test.txt")
test_x = insert_weight_one(test_X)
t_y = predict(test_x, theta)
mse_test = mean_square_error(test_y, t_y)

print(f'theta: \n{theta}')
print(f'mse_training: {mse}, mse_test: {mse_test}')
