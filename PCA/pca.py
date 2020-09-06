import numpy as np
import matplotlib.pyplot as plt


def center_mean(array):
    return array - array.mean()


def covariance_matrix(matrix):
    return np.dot(matrix.T, matrix)


def compute_eigen(reshaped_matrix):
    return np.linalg.eig(reshaped_matrix)


def select_eigen(eigen_val, eigen_vecs):
    index = -1
    max_val = max(eigen_val)
    for i in range(len(eigen_val)):
        if eigen_val[i] == max_val:
            index = i
    vec = eigen_vecs[index]
    normalised_eigen_vec = vec / np.sqrt((np.linalg.norm(vec)))
    return max_val, normalised_eigen_vec


col_1, col_2 = np.loadtxt("pca_toy.txt", unpack=True)
mc_col_1, mc_col_2 = center_mean(col_1), center_mean(col_2)
matrix = np.vstack((mc_col_1, mc_col_2))
cov_matrix = covariance_matrix(np.vstack((mc_col_1, mc_col_2)).T)
print(f'Covariance Matrix: {cov_matrix}, \nShape: {cov_matrix.shape}')
eigen_value, eigen_vectors = compute_eigen(cov_matrix)
print(f'Eigen Value: {eigen_value}, Eigen Vector: {eigen_vectors}')
max_eigen_val, max_eigen_vector = select_eigen(eigen_value, eigen_vectors.T)

Z = np.dot(max_eigen_vector, matrix) * max_eigen_val
z_x = Z * max_eigen_vector[0]
z_y = Z * max_eigen_vector[1]

origin = [0, 0]
plt.scatter(col_1, col_2, color=['b'])
plt.scatter(mc_col_1, mc_col_2, color=['r'])
plt.quiver(*origin, *eigen_vectors[:, 0], color=['g'], scale=4)
plt.quiver(*origin, *eigen_vectors[:, 1], color=['g'], scale=4)
plt.scatter(z_x, z_y, color=['g'])
plt.show()
