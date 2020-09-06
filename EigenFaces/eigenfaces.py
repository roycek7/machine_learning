"""
@author: s4625266
"""

import math
import sys

import matplotlib.pyplot as plt
import numpy as np

pixel = (32, 32)  # pixels
plt.figure(figsize=(5, 5))  # figure size
color_scale = 'gray'  # gray scale
image_rotation = 3  # rotate 270 degrees

KNN = 1  # k nearest neighbour
arg_req, top_faces, min_K = 5, 5, 5  # arguments required, top 5 eigenvectors, minimum value of K


def mean_face(array, plot_img=False) -> np.ndarray:
    """mean face of the training set and plot image if flagged"""
    m_f = array.mean(axis=1)
    if plot_img:
        plot_image(rotate_image(np.resize(m_f, pixel)), title='Mean Face')
    return m_f


def centre_mean(array) -> np.ndarray:
    """subtract mean from the training set"""
    return array - array.mean(axis=0)


def compute_covariance(matrix) -> np.ndarray:
    """covariance matrix of dataset"""
    return np.dot(matrix.T, matrix)


def compute_eig_vector_values(cov_matrix) -> np.array:
    """returns sorted eigenvalue and eigenvectors in ascending order"""
    return np.linalg.eigh(cov_matrix)


def compute_dot_product(x, y) -> np.ndarray:
    """returns dot product between two entity"""
    return np.dot(x, y)


def rotate_image(vector) -> np.ndarray:
    """rotate image three times"""
    return np.rot90(vector, image_rotation)


def k_eig_vectors(eig_val, eig_vec, k_th) -> np.ndarray:
    """returns sorted and the K largest normalised eigen vectors"""
    eig_vectors = eig_vec[:, eig_val.argsort()[::-1]]
    for i in range(len(eig_val)):
        eig_vec[i] /= np.linalg.norm(eig_vec[i])
    return eig_vectors[:, :k_th]


def sort_tuple(tup, y) -> set:
    """returns sorted tuple"""
    tup.sort(key=lambda x: x[y])
    return tup


def euclidean_distance(test_data_vector, training_data_vector) -> float:
    """calculates the euclidean distance between two vectors"""
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(test_data_vector, training_data_vector)]))


def reconstruct_image(vector, m_f, top_ev, k, weight=None, weights_calculated=False) -> np.ndarray:
    """reconstruct a vector with a compact formula if not flagged else with weights"""
    if weights_calculated:
        return compute_dot_product(weight, top_ev) + m_f
    reconstructed_image = m_f + compute_dot_product(compute_dot_product(vector - m_f, top_ev), top_ev.T)
    plot_image(rotate_image(np.resize(reconstructed_image, pixel)), title=f'Reconstructed Test Image for K: {k}')


def accuracy_metric(actual, predicted, correct=0) -> float:
    """calculate accuracy of the dataset comparing the actual test label vs the predicted label"""
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / len(actual) * 100.0


def plot_image(image, image_vector=None, subplot=False, horizontal=False, title=''):
    """plot image individual or subplots based on flags"""
    if not subplot:
        plt.imshow(image, cmap=color_scale)
        plt.axis('off')
        plt.title(title)
        plt.show()
    else:
        for i in range(len(image) if not horizontal else top_faces):
            if horizontal:
                image_vector = rotate_image(np.resize(image[:, i], pixel))
            plt.subplot(1 if horizontal else len(image), 1 if not horizontal else top_faces, i + 1)
            plt.imshow(image[i] if not horizontal else image_vector, cmap=color_scale)
            plt.axis('off')
        plt.title(title)
        plt.show()


def plot_requirements(training_data, test_data, k, eig_value, eig_vector, train_label,
                      test_label, top_eig_vectors, incorrect_images, threshold):
    """plots the mean face of the training set, top 5 eigen-face, 4 images of test_data for different K and
    the plot of 1NN classification rate upto k"""
    # # requirement 1
    mean_face(training_data.T, plot_img=True), plot_image(top_eig_vectors, subplot=True, horizontal=True,
                                                          title='Top 5 EigenFaces')

    # # requirement 2
    # reconstruction function with different k_th value, test_data[0] : first image from test dataset
    [reconstruct_image(test_data[0], mean_face(training_data.T),
                       k_eig_vectors(eig_value, eig_vector, k_th=i), i) for i in range(10, 120, 30)]

    # # requirement 3
    print(f'Computing plot for the nearest-neighbour (1NN) classification rate of K: {k}')
    plt.plot(range(k - 1), [knn_classifier(training_data, test_data, mean_face(training_data.T),
                                           k_eig_vectors(eig_value, eig_vector, i),
                                           train_label, test_label, threshold, project=True)[0]
                            for i in range(1, k)])
    plt.title(f'1NN classification rate of K: {k}')
    plt.xlabel('K')
    plt.ylabel('Accuracy')
    plt.show()

    # # requirement 4
    plot_image(incorrect_images, subplot=True, title=f'Incorrect Classified Faces for K: {k}')


def knn_classifier(train_images, test_images, mean_f, top_eig_vectors,
                   train_label, test_label, threshold, print_statement=False, project=False) -> tuple:
    """finds the euclidean distance between reconstructed image and the test image, calculates the 1NN of the
    eigen-face and returns the accuracy, incorrect faces vectors and non-face rate"""
    no_face_detected = 0
    predicted, incorrect_images = [], []
    projected = [compute_dot_product(img_vec - mean_f, top_eig_vectors) for img_vec in train_images]
    for i in range(len(test_images)):
        distance = []
        for j in range(len(projected)):
            eu_distance = euclidean_distance(reconstruct_image(0, mean_f, top_eig_vectors.T, 0, projected[j],
                                                               weights_calculated=True),
                                             test_images[i]) if not project else \
                euclidean_distance(projected[j], compute_dot_product(test_images[i] - mean_f, top_eig_vectors))
            distance.append((eu_distance, train_label[j]))
        minimum_distant_neighbours = sort_tuple(distance, 0)[:KNN]

        nearest_neighbour, reconstruct_error = [], []
        for j in minimum_distant_neighbours:
            reconstruct_error.append(j[0]), nearest_neighbour.append(j[-1])

        predicted.append(max(nearest_neighbour, key=nearest_neighbour.count))

        if print_statement:
            print(f'({i + 1}) ~ TEST IMAGE LABEL: {int(test_label[i])} | '
                  f'RECONSTRUCTION ERROR: {round(reconstruct_error[0], 2)} | '
                  f'CLASSIFIED TRAINING LABEL: {int(predicted[i])}')

        if reconstruct_error[0] > threshold:
            no_face_detected += 1
            if print_statement:
                print('-----------------------------------------')
                print(f'| TEST IMAGE: {int(test_label[i])} CLASSIFIED @ NON-FACE |')
                print('-----------------------------------------')

        if test_label[i] != predicted[i]:
            incorrect_images.append(
                np.concatenate(
                    (rotate_image(np.resize(test_images[i], pixel)),
                     rotate_image(np.resize(train_images[int(predicted[i])], pixel))),
                    axis=1))

    non_face_rate = 100 * no_face_detected / len(test_images)
    return round(accuracy_metric(test_label, predicted), 2), incorrect_images, round(non_face_rate, 2)


def eig_faces(argv):
    """eigen face function to load training data, training label, number of principal components, test data, test label
    data, calls function to compute the top eigen vectors and classify images"""
    try:
        if len(argv) < arg_req:
            raise Exception
        else:
            training_data, training_data_label, K, test_data, test_data_label = np.genfromtxt(argv[0]), \
                                                                                np.genfromtxt(argv[1]), int(
                argv[2]), np.genfromtxt(argv[3]), np.genfromtxt(argv[4])

            if K < min_K:
                print(f'Minimum value of K should be greater than or equal to {min_K}'.upper())
                raise Exception

            eig_value, eig_vector = compute_eig_vector_values(compute_covariance(centre_mean(training_data)))
            top_eig_vectors = k_eig_vectors(eig_value, eig_vector, K)

            threshold = 1115
            accuracy, incorrect_images, non_face_rate = knn_classifier(training_data, test_data,
                                                                       mean_face(training_data.T),
                                                                       top_eig_vectors, training_data_label,
                                                                       test_data_label, threshold,
                                                                       print_statement=True)

            print(f'Accuracy: {accuracy}%\nNon-Face Rate: {non_face_rate}%')

            plot_requirements(training_data, test_data, K, eig_value, eig_vector, training_data_label,
                              test_data_label, top_eig_vectors, incorrect_images, threshold)
    except Exception as e:
        print(f'{str(e).upper()}\nFollow Command:\npython eigenfacesassignment.py faces_train.txt '
              f'faces_train_labels.txt 10 faces_test.txt faces_test_labels.txt')


if __name__ == '__main__':
    eig_faces(sys.argv[1:])
