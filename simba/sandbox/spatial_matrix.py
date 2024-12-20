import numpy as np
import pysal
from pysal.explore import esda


def create_adjacency_matrix(N, M):

    def idx(r, c, M):
        return r * M + c
    adj_matrix = np.zeros((N * M, N * M))

    for r in range(N):
        for c in range(M):
            current = idx(r, c, M)
            if r > 0:
                adj_matrix[current, idx(r - 1, c, M)] = 1
            if r < N - 1:
                adj_matrix[current, idx(r + 1, c, M)] = 1
            if c > 0:
                adj_matrix[current, idx(r, c - 1, M)] = 1
            if c < M - 1:
                adj_matrix[current, idx(r, c + 1, M)] = 1
            if r > 0 and c > 0:
                adj_matrix[current, idx(r - 1, c - 1, M)] = 1
            if r > 0 and c < M - 1:
                adj_matrix[current, idx(r - 1, c + 1, M)] = 1
            if r < N - 1 and c > 0:
                adj_matrix[current, idx(r + 1, c - 1, M)] = 1
            if r < N - 1 and c < M - 1:
                adj_matrix[current, idx(r + 1, c + 1, M)] = 1

    return adj_matrix


def morans_i(values, adj_matrix):
    """Calculate Moran's I for spatial autocorrelation."""
    n = len(values)
    mean_value = np.mean(values)  # Mean of the values
    num = 0
    denom = 0
    W = np.sum(adj_matrix)  # Sum of weights in the adjacency matrix

    for i in range(n):
        for j in range(n):
            if adj_matrix[i, j] == 1:  # If i and j are neighbors
                num += (values[i] - mean_value) * (values[j] - mean_value)
        denom += (values[i] - mean_value) ** 2

    I = (n / W) * (num / denom)  # Moran's I
    return I

# Example grid size (e.g., 3x3 grid)
N, M = 3, 3

# Example 2D grid values for each cell in the grid
values = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Flatten the 2D grid into 1D array for Moran's I computation
flattened_values = values.flatten()

adj_matrix = create_adjacency_matrix(N, M)

I = morans_i(flattened_values, adj_matrix)