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
def spawn_walker(M):
    """Spawn single walker at top random x position."""
    return 0, np.random.randint(0, M)

@njit
def monte_carlo_dla_ps(grid_size, steps, ps):
    np.random.seed(10)
    N, M = grid_size
    cluster = np.zeros((N, M), dtype=np.uint8)
    cluster[N - 1, M // 2] = 1  # Initial seed bottom center

    moves = np.array([(1, 0), (-1, 0), (0, 1), (0, -1)])  

    y, x = spawn_walker(M)  # one walker

    for _ in range(steps):
        move = moves[np.random.randint(0, 4)]  
        y += move[0]
        x = (x + move[1]) % M  # Periodic boundary in x

        if y >= N or y < 0:
            y, x = spawn_walker(M)  # Respawn at top
            continue  # Start moving new walker

        # Check if walker sticks
        if is_next_to_cluster(y, x, cluster) and np.random.rand() <= ps:
            cluster[y, x] = 1  
            y, x = spawn_walker(M)  # Spawn new walker
            continue  # Start moving the new walker

    return cluster
