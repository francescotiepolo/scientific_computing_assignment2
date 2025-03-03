import numpy as np
import matplotlib.pyplot as plt
from numba import njit

@njit
def is_next_to_cluster(y, x, cluster):
    N, M = cluster.shape
    moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for dy, dx in moves:
        ny, nx = y + dy, (x + dx) % M  # periodic in x-direction
        if 0 <= ny < N and cluster[ny, nx]:  # within y range
            return True
    return False

@njit
def spawn_walkers(num_walkers, M):
    walkers_y = np.zeros(num_walkers, dtype=np.int32)  # Spawn at top (y=0)
    walkers_x = np.random.randint(0, M, num_walkers)  # Random x position
    return walkers_y, walkers_x

@njit
def monte_carlo_dla(grid_size, num_walkers, steps):
    N, M = grid_size
    cluster = np.zeros((N, M), dtype=np.uint8)
    cluster[N - 1, M // 2] = 1  # Initial seed at bottom center

    moves = np.array([(1, 0), (-1, 0), (0, 1), (0, -1)]) 

    walkers_y, walkers_x = spawn_walkers(num_walkers, M)  # Initialize walkers

    for _ in range(steps):
        random_moves = moves[np.random.randint(0, 4, num_walkers)]
        walkers_y += random_moves[:, 0] 
        walkers_x += random_moves[:, 1]  

        # Periodic boundary conditions in x-direction
        walkers_x = np.mod(walkers_x, M)

        # Respawn walkers at top if they hit the top or bottom boundary
        for i in range(num_walkers):
            if walkers_y[i] >= N or walkers_y[i] < 0:  
                walkers_y[i] = new_y[0]  # Assign separately to avoid numba error
                walkers_x[i] = new_x[0]

        # Check which walkers should stick to the cluster
        for i in range(num_walkers):
            y, x = walkers_y[i], walkers_x[i]
            if is_next_to_cluster(y, x, cluster):
                cluster[y, x] = 1  # Walker sticks to the cluster
                
                new_y, new_x = spawn_walkers(1, M)  
                walkers_y[i] = new_y[0]  
                walkers_x[i] = new_x[0]

    return cluster
