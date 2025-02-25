import numpy as np
import matplotlib.pyplot as plt
from numba import njit

@njit
def is_next_to_cluster(y, x, cluster):
    N, M = cluster.shape
    moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]  
    for dy, dx in moves:
        ny, nx = y + dy, x + dx
        if 0 <= ny < N and 0 <= nx < M and cluster[ny, nx]:
            return True
    return False

@njit
def monte_carlo_dla(grid_size, steps):
    N, M = grid_size
    cluster = np.zeros((N, M), dtype=np.uint8)
    cluster[N // 2, M // 2] = 1  # Initial seed at center

    moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]  

    for _ in range(steps):
        # Start walker at a random x position on the top row (y = 0)
        x = np.random.randint(0, M)
        y = N  

        while True:
            dy, dx = moves[np.random.randint(0, 4)]
            y, x = y + dy, x + dx

            #  periodic boundary conditions 
            if x < 0:
                x = M - 1
            elif x >= M:
                x = 0

            # if walker exits from top or bottom it stops existing
            if y < 0 or y >= N:
                break  

            # Walker sticks if touches cluster 
            if is_next_to_cluster(y, x, cluster):
                cluster[y, x] = 1
                break  

    return cluster