from numpy.linalg import norm
from numpy import dot
import numpy as np


def compute_vector_length(vector):
    len_of_vector = np.linalg.norm(vector)

    return len_of_vector


vector = np.array([-2, 4, 9, 21])
result = compute_vector_length([vector])
print(round(result, 2))


def compute_dot_product(vector1, vector2):
    result = np.dot(vector1, vector2)

    return result


v1 = np.array([0, 1, -1, 2])
v2 = np.array([2, 5, 1, 0])
result = compute_dot_product(v1, v2)

print(round(result, 2))


def matrix_multi_vector(matrix, vector):
    result = np.dot(matrix, vector)
    return result


m = np.array([[-1, 1, 1], [0, -4, 9]])
v = np.array([0, 2, 1])
result = matrix_multi_vector(m, v)
print(result)


def matrix_multi_matrix(matrix1, matrix2):
    result = np.dot(matrix1, matrix2)
    return result


m1 = np.array([[0, 1, 2], [2, -3, 1]])
m2 = np.array([[1, -3], [6, 1], [0, -1]])
result = matrix_multi_matrix(m1, m2)
print(result)

m1 = np.eye(3)
m2 = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
result = m1@m2
print(result)


def inverse_matrix(matrix):
    result = np.linalg.inv(matrix)
    return result


m1 = np.array([[-2, 6], [8, -4]])
result = inverse_matrix(m1)
print(result)


def compute_eigenvalues_eigenvectors(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    return eigenvalues, eigenvectors


matrix = np.array([[0.9, 0.2], [0.1, 0.8]])
eigenvalues, eigenvectors = compute_eigenvalues_eigenvectors(matrix)
print(eigenvectors)


def compute_cosine(v1, v2):
    cos_sim = compute_dot_product(
        v1, v2) / (compute_vector_length(v1)*compute_vector_length(v2))

    return cos_sim
