import numpy as np
from numba import njit, prange

@njit
def init_concentration(N, M):
    ''' Initialize the concentration field (in the equilibrium state to speed up simulation)
    Inputs:
    - N: int, number of rows
    - M: int, number of columns
    Outputs:
    - c: np.array, concentration field/matrix
    '''
    y = np.linspace(0, 1, N) # Create a linear space from 0 to 1 with N points
    c = np.zeros((N, M))
    for j in range(M): # Copy the y values to all columns
        c[:, j] = y
    return c

@njit(parallel=True)
def solve_laplace(c, cluster, w=1.8, tol=1e-5, max_iter=1000):
    ''' Solve the Laplace equation using the SOR method
    Inputs:
    - c: np.array, concentration field/matrix
    - cluster: np.array, binary matrix indicating the cluster
    - w: float, relaxation parameter
    - tol: float, tolerance parameter to stop iteration
    - max_iter: int, maximum number of iterations
    Outputs:
    - c: np.array, concentration field/matrix after solving the Laplace equation
    '''
    N, M = c.shape
    for iter in range(max_iter): # Loop until max_iter if tolerance is not reached
        old_c = c.copy() # Save old field to compute new one
        diff = 0.0 # Initialize the difference, later compared to tolerance
        for i in prange(1, N-1): # Loop over every cell (parallelized)
            for j in range(1, M-1):
                if not cluster[i, j]: # If the cell is not part of the cluster, compute new value, else it is not needed as it is set equal to 0
                    c[i, j] = max((1 - w) * old_c[i, j] + 
                                  w * 0.25 * (old_c[i+1, j] + 
                                              c[i-1, j] + 
                                              old_c[i, j+1] + 
                                              c[i, j-1]), 0) # Make sure the concentration field is positive (negative values might result from floting-point precision erros and propagation of erros)
                    diff = max(diff, abs(c[i, j] - old_c[i, j])) # Store the largest difference between new and old grid
        if diff < tol: # Stop the loop when the tolerance is reached
            break
    return c 
@njit
def growth_candidates(cluster):
    ''' Find the growth candidates for the cluster
    Inputs:
    - cluster: np.array, binary matrix indicating the cluster
    Outputs:
    - candidates: np.array, list of coordinates of the growth candidates
    '''
    N, M = cluster.shape
    candidates = [] # Initialize the list of candidates
    for i in range(N): # Loop over every cell
        for j in range(M):
            if not cluster[i, j]: # If the cell is not part of the cluster, check if it has a neighbor in the cluster
                neighbors = []
                if i > 0: # Check if the north neighbor is in the cluster
                    neighbors.append(cluster[i-1, j])
                if i < N-1: # Check if the south neighbor is in the cluster
                    neighbors.append(cluster[i+1, j])
                if j > 0: # Check if the west neighbor is in the cluster
                    neighbors.append(cluster[i, j-1])
                if j < M-1: # Check if the east neighbor is in the cluster
                    neighbors.append(cluster[i, j+1])
                if np.sum(np.array(neighbors)) > 0: # If at least one neighbor is in the cluster, add the cell to the candidates
                    candidates.append((i, j))
    return np.array(candidates, dtype=np.int64).reshape(-1, 2) # Return the candidates as a numpy array with 2 columns (i, j)

@njit
def choose_candidate(candidates, prob):
    ''' Choose a candidate to add to the cluster based on probabilities (here comulative to adapt to numba)
    Inputs:
    - candidates: np.array, list of coordinates of the growth candidates
    - prob: np.array, list of probabilities for each candidate
    Outputs:
    - i_c: int, row index of the chosen candidate
    - j_c: int, column index of the chosen candidate
    '''
    cumulative_prob = np.cumsum(prob) # Compute the cumulative probabilities
    rand = np.random.rand() # Generate a random number between 0 and 1
    for i, cp in enumerate(cumulative_prob): # Loop over the cumulative probabilities
        if rand < cp: # If the random number is smaller than the cumulative probability, choose the candidate
            return (candidates[i, 0], candidates[i, 1])
    return (candidates[-1, 0], candidates[-1, 1])

@njit(parallel=True)
def simulation_dla(grid_size=(100, 100), steps=500, eta=1.0, w=1.8):
    ''' Perform Diffusion Limited Aggregation (DLA) simulation
    Inputs:
    - grid_size: tuple, size of the grid (N, M)
    - steps: int, number of steps for the simulation
    - eta: float, parameter for shape of the object
    - w: float, relaxation parameter for solve_laplace
    Outputs:
    - c: np.array, concentration field/matrix after the simulation
    - cluster: np.array, binary matrix indicating the cluster
    - history: list, list of coordinates of the added cells
    '''
    N, M = grid_size
    cluster = np.zeros((N, M), dtype=np.uint8) # Initialize the cluster
    cluster[0, M//2] = True # Seed the cluster in the middle of the bottom row
    c = init_concentration(N, M) # Initialize the concentration field
    for i in range(N): # Set the concentration to 0 for the cluster cells
        for j in range(M):
            if cluster[i, j]: 
                c[i, j] = 0.0
    history = [] # Initialize the history of added cells
    for step in range(steps): # Loop over the number of steps
        c = solve_laplace(c, cluster, w=w, tol=1e-5, max_iter=1000) # Solve the Laplace equation according to the present cluster configuration
        candidates = growth_candidates(cluster) # Find the growth candidates
        if candidates.size ==0: # If there are no candidates, stop the simulation
            break
        weights = np.array([max(c[i, j], 0)**eta for (i, j) in candidates]) # Compute the weights based on the concentration field
        weights = np.clip(weights, 0, None) # Make sure the weights are positive
        total_weight = np.sum(weights) # Compute the total weight for narmalization
        prob = weights / total_weight # Compute the probabilities

        i_c, j_c = choose_candidate(candidates, prob) # Choose a candidate based on the probabilities
        cluster[i_c, j_c] = 1 # Add the chosen candidate to the cluster

        c[i_c, j_c] = 0.0 # Set the concentration to 0 for the added cell
        history.append((i_c, j_c)) # Add the added cell to the history
    return c, cluster, history
