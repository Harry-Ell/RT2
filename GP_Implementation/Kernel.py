'''
Kernel functional form  lifted from thesis by Niklas Wahlström

Attempted implementation of his kernel function. 
'''

import numpy as np 


def curl_free_kernel(x1: np.ndarray, x2: np.ndarray, sigma_f: float = 1, l: float = 1) -> np.ndarray:
    """
    Divergence-free kernel based on a squared-exponential base.
    returns a 3x3 covariance matrix for vector-valued outputs in R^3.
    
    formula 2.44
    """
    diff = x1 - x2
    r2 = np.dot(diff, diff)
    scaling = (sigma_f**2 / l**2) * np.exp(- r2 / (2 * l**2))
    K = scaling * (np.eye(len(x1)) - np.outer(diff, diff) / l**2)
    return K



def divergence_free_kernel(x1:np.array, x2:np.array, sigma_f:float =1, l:float=1)->np.ndarray:
    '''
    Should be a divergence and curl free vector field which you get from here. There surely arent that 
    many of these, so this will hopefully quickly look like a magnetic field. 

    formula 2.48
    '''
    diff = x1 - x2
    # def some distance measure
    r2 = np.dot(diff, diff)
    term1 = (2 - r2 / l**2) * np.eye(len(x1))
    term2 = np.outer(diff, diff) / l**2
    scaling = np.exp(- r2 / (2 * l**2)) * (sigma_f / l)**2
    K = (term1 + term2) * scaling
    return K



def sample_vector_field(grid_x: np.ndarray, grid_y: np.ndarray, grid_z: np.ndarray,
                        kernel_func:str, sigma_f: float = 1, l: float = 1, random_seed: int = 1)->np.ndarray:
    """
    Given 3D meshgrid arrays, build the full cov matrix for a vector field 
    using the inputted kernel function.
    
    Returns the grid x,y,z and the sampled field components U,V,W shaped as the grid, 
    just so you can drop it straight into the plotter code.
    """
    np.random.seed(random_seed)
    # some hacky logic 
    function_chosen = divergence_free_kernel if kernel_func == 'divergence_free_kernel' else curl_free_kernel

    points = np.vstack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()]).T
    N = points.shape[0]  # total number of spatial points
    D = 3  # dimensionality

    # Build the full covariance matrix (3N x 3N)
    cov = np.zeros((N * D, N * D))

    for i in range(N):
        for j in range(N):
            K_ij = function_chosen(points[i], points[j], sigma_f=sigma_f, l=l)  # (3,3)
            cov[i*D:(i+1)*D, j*D:(j+1)*D] = K_ij

    # generate a sample from a mean 0 field
    mean = np.zeros(N * D)
    sample = np.random.multivariate_normal(mean, cov)

    # reshape for plotting purposes    
    field_vectors = sample.reshape(N, D)   
    U = field_vectors[:, 0].reshape(grid_x.shape)
    V = field_vectors[:, 1].reshape(grid_y.shape)
    W = field_vectors[:, 2].reshape(grid_z.shape)
    
    return grid_x, grid_y, grid_z, U, V, W


def updated_vector_field(sampled_points_x: np.ndarray, sampled_points_y: np.ndarray, sampled_points_z: np.ndarray,
                         sampled_fields_x: np.ndarray, sampled_fields_y: np.ndarray, sampled_fields_z: np.ndarray,
                         grid_x: np.ndarray, grid_y: np.ndarray, grid_z: np.ndarray,
                         kernel_func: str, sigma_f: float = 1, l: float = 1, random_seed: int = 1, 
                         noise:float = 1e-3):
    """
    Compute the GP-posterior vector field given noisy vector observations at a few points.
    returns a 3D vector field evaluated over the entire grid, conditioned on known field measurements.
    """

    np.random.seed(random_seed)

    # Choose kernel
    function_chosen = divergence_free_kernel if kernel_func == 'divergence_free_kernel' else curl_free_kernel

    # this allows us to loop over these points
    points = np.vstack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()]).T               # (N, 3)
    points_sampled = np.vstack([sampled_points_x.flatten(), sampled_points_y.flatten(), sampled_points_z.flatten()]).T  # (M, 3)
    fields_sampled = np.vstack([sampled_fields_x.flatten(), sampled_fields_y.flatten(), sampled_fields_z.flatten()]).T  # (M, 3)

    N = points.shape[0]           # number of grid points
    M = points_sampled.shape[0]   # number of observations
    D = 3                         # vector field dimensionality

    # Allocate covariance matrices
    K_x_x = np.zeros((M * D, M * D))        # covariance between sampled points
    K_xstar_x = np.zeros((N * D, M * D))    # covariance between grid and sampled
    K_xstar_xstar = np.zeros((N * D, N * D))  # covariance between grid points
    # also our observations vector 
    y = np.zeros(M * D)

    # Build K_x_x
    for i in range(M):
        for j in range(M):
            K_ij = function_chosen(points_sampled[i], points_sampled[j], sigma_f=sigma_f, l=l)  # (3x3)
            K_x_x[i*D:(i+1)*D, j*D:(j+1)*D] = K_ij

    # Build K_xstar_x
    for i in range(N):
        for j in range(M):
            K_ij = function_chosen(points[i], points_sampled[j], sigma_f=sigma_f, l=l)  # (3x3)
            K_xstar_x[i*D:(i+1)*D, j*D:(j+1)*D] = K_ij

    # Build K_xstar_xstar
    for i in range(N):
        for j in range(N):
            K_ij = function_chosen(points[i], points[j], sigma_f=sigma_f, l=l)  # (3x3)
            K_xstar_xstar[i*D:(i+1)*D, j*D:(j+1)*D] = K_ij

    # Build y
    for i in range(M):
        y[i*D:(i+1)*D] = fields_sampled[i]  # (3x,)?

    # GP posterior mean and cov sampled
    K_x_x = K_x_x + noise * np.eye(M * D)
    posterior_mean = K_xstar_x @ np.linalg.solve(K_x_x, y)
    posterior_covar = K_xstar_xstar - K_xstar_x @ np.linalg.solve(K_x_x, K_xstar_x.T)
    sample = np.random.multivariate_normal(posterior_mean, posterior_covar)

    # Reshape into 3D vector field
    field_vectors = sample.reshape(N, D)
    U = field_vectors[:, 0].reshape(grid_x.shape)
    V = field_vectors[:, 1].reshape(grid_y.shape)
    W = field_vectors[:, 2].reshape(grid_z.shape)

    return grid_x, grid_y, grid_z, U, V, W


def aggregated_hyperparameter_optimisation(sampled_points_x: np.ndarray, sampled_points_y: np.ndarray, sampled_points_z: np.ndarray,
                                sampled_fields_x: np.ndarray, sampled_fields_y: np.ndarray, sampled_fields_z: np.ndarray,
                                grid_x: np.ndarray, grid_y: np.ndarray, grid_z: np.ndarray)->list[float, float]:
    '''
    hyperparameter optimisation procedure for the case of tuning both sigma_f and the length scale. 

    Perhaps also the noise/ jitter term could be incorporated into this?
    '''
    # Optimise the length scale:
    
    # Optimise whatever sigma_f is 

    # Optimise the noise:


    pass

def length_scale_optimiser