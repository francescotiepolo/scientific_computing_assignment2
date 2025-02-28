import numpy as np
import matplotlib.pyplot as plt
from numba import njit

@njit
def is_next_to_cluster(y, x, cluster):
    N, M = cluster.shape
    moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for dy, dx in moves:
        ny, nx = y + dy, (x + dx) % M  # in x
        if 0 <= ny < N and cluster[ny, nx]:  # in y range
            return True
    return False

@njit
def spawn_walkers(num_walkers, M):
    walkers_y = np.zeros(num_walkers, dtype=np.int32)  # start at y = 0
    walkers_x = np.random.randint(0, M, num_walkers)  # Random x 
    return walkers_y, walkers_x

@njit
def monte_carlo_dla(grid_size, num_walkers, steps):
    N, M = grid_size
    cluster = np.zeros((N, M), dtype=np.uint8)
    cluster[N - 1, M // 2] = 1  

    moves = np.array([(1, 0), (-1, 0), (0, 1), (0, -1)]) 

    walkers_y, walkers_x = spawn_walkers(num_walkers, M)  # Initial walkers

    for _ in range(steps):
        random_moves = moves[np.random.randint(0, 4, num_walkers)]
        walkers_y += random_moves[:, 0] 
        walkers_x += random_moves[:, 1]  

        # periodic boundary conditions
        walkers_x = np.mod(walkers_x, M)

        # fixed boundary conditions 
        walkers_y = np.clip(walkers_y, 0, N - 1)  

        # check which walkers should stick to the cluster
        for i in range(num_walkers):
            y, x = walkers_y[i], walkers_x[i]
            if is_next_to_cluster(y, x, cluster):
                cluster[y, x] = 1  # Walker sticks to the cluster
                
                new_y, new_x = spawn_walkers(1, M)  
                walkers_y[i] = new_y[0]  
                walkers_x[i] = new_x[0]  

    return cluster