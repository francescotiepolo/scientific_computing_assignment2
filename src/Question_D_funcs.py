import numpy as np

def initialize_grid(size, noise=0.02):
    u = np.ones((size, size)) * 0.5
    v = np.zeros((size, size))
    
    #small square in the center with v = 0.25
    r = size // 10  #region size
    cx, cy = size // 2, size // 2
    v[cx - r:cx + r, cy - r:cy + r] = 0.25
    
    #add some noise
    u += noise * np.random.rand(size, size)
    v += noise * np.random.rand(size, size)
    
    return u, v

def laplacian(Z):
    return (
        np.roll(Z, 1, axis=0) + np.roll(Z, -1, axis=0) +
        np.roll(Z, 1, axis=1) + np.roll(Z, -1, axis=1) -
        4 * Z
    )

def update(u, v, Du, Dv, f, k, dt, dx, noise=0.0):
    Lu = laplacian(u) / dx**2
    Lv = laplacian(v) / dx**2
    
    uvv = u * v**2
    u += (Du * Lu - uvv + f * (1 - u)) * dt + noise * np.random.rand(*u.shape)
    v += (Dv * Lv + uvv - (f + k) * v) * dt + noise * np.random.rand(*v.shape)
    
    return u, v

def run_simulation(size, steps, parameter_sets, dx, dt, noise=0.0):
    results = []
    for Du, Dv, f, k in parameter_sets:
        u, v = initialize_grid(size, noise=noise)
        for _ in range(steps):
            u, v = update(u, v, Du, Dv, f, k, dt, dx, noise=noise)
        results.append((u, v, Du, Dv, f, k, noise))
    return results





