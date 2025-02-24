import numpy as np
from numba import njit

@njit
def init_concentration(N, M):
    y = np.linspace(0, 1, N)
    c = np.zeros((N, M))
    for j in range(M):
        c[:, j] = y
    return c

@njit
def solve_laplace(c, cluster, w=1.8, tol=1e-5, max_iter=1000):
    N, M = c.shape
    for iter in range(max_iter):
        old_c = c.copy()
        diff = 0.0
        for i in range(1, N-1):
            for j in range(1, M-1):
                if not cluster[i, j]:
                    c[i, j] = (1 - w) * old_c[i, j] + w * 0.25 * (old_c[i+1, j] + c[i-1, j] + old_c[i, j+1] + c[i, j-1])
                    diff = max(diff, abs(c[i, j] - old_c[i, j]))
        if diff < tol:
            break
    return c

@njit
def growth_candidates(cluster):
    N, M = cluster.shape
    candidates = []
    for i in range(N):
        for j in range(M):
            if not cluster[i, j]:
                neighbors = []
                if i > 0:
                    neighbors.append(cluster[i-1, j])
                if i < N-1:
                    neighbors.append(cluster[i+1, j])
                if j > 0:    
                    neighbors.append(cluster[i, j-1])
                if j < M-1:
                    neighbors.append(cluster[i, j+1])
                if np.sum(np.array(neighbors)) > 0:
                    candidates.append((i, j))
    return np.array(candidates, dtype=np.int64).reshape(-1, 2)

@njit
def choose_candidate(candidates, prob):
        cumulative_prob = np.cumsum(prob)
        rand = np.random.rand()
        for i, cp in enumerate(cumulative_prob):
            if rand < cp:
                return (candidates[i, 0], candidates[i, 1])
        return (candidates[-1, 0], candidates[-1, 1])

@njit
def simulation_dla(grid_size=(100, 100), steps=500, eta=1.0, w=1.8):
    N, M = grid_size
    cluster = np.zeros((N, M), dtype=np.uint8)
    cluster[0, M//2] = True
    c = init_concentration(N, M)
    for i in range(N):
        for j in range(M):
            if cluster[i, j]: 
                c[i, j] = 0.0
    history = []
    for step in range(steps):
        c = solve_laplace(c, cluster, w=w, tol=1e-5, max_iter=1000)
        candidates = growth_candidates(cluster)
        if candidates.size ==0:
            break
        weights = np.array([c[i, j]**eta for (i, j) in candidates])
        weights = np.clip(weights, 0, None)
        total_weight = np.sum(weights)
        prob = weights / total_weight

        i_c, j_c = choose_candidate(candidates, prob)
        cluster[i_c, j_c] = 1

        c[i_c, j_c] = 0.0
        history.append((i_c, j_c))
    return c, cluster, history
