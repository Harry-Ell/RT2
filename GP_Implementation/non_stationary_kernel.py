'''
swing at doing this non spatially homogenous thing. 

we need some sort of recursion:
1. fit the gp as usual
2. use this to determine the function from which we transform space. 
3. refit the gp using this new scale and see what happens, could recurse again but we may approach overfitting
'''

import numpy as np 
from scipy.optimize import minimize



def divergence_free_kernel(x1:np.array, x2:np.array, sigma_f:float =1, l:float=1)->np.ndarray:
    '''
    Should be a divergence free vector field which you get from here. There surely arent that 
    many of these, so this will hopefully quickly look like a magnetic field. 

    formula 2.48
    '''
    
    # if np.any((np.sqrt(x1[0]**2+x1[1]**2) < 0.2) and (np.sqrt(x2[0]**2+x2[1]**2) < 0.2)):
    #     l = 0.2
    # elif (np.sqrt(x1[0]**2+x1[1]**2) < 0.2) or (np.sqrt(x2[0]**2+x2[1]**2) < 0.2):
    #     return np.zeros((3,3))
    # else:
    #     l = 1
    diff = x1 - x2
    # def some distance measure
    r2 = np.dot(diff, diff)
    term1 = (2 - r2 / l**2) * np.eye(len(x1))
    term2 = np.outer(diff, diff) / l**2
    scaling = np.exp(- r2 / (2 * l**2)) * (sigma_f / l)**2
    K = (term1 + term2) * scaling
    return K




def sample_vector_field(kernel_func:str, sigma_f: float = 1, l: float = 1, random_seed: int = 1)->np.ndarray:
    """
    could be interesting to keep and see what the field spits out 

    Given 3D meshgrid arrays, build the full cov matrix for a vector field 
    using the inputted kernel function.
    
    Returns the grid x,y,z and the sampled field components U,V,W shaped as the grid, 
    just so you can drop it straight into the plotter code.
    """
    grid_x, grid_y, grid_z = np.meshgrid(
    np.linspace(-0.5, 0.5, 10),
    np.linspace(-0.5, 0.5, 10),
    np.linspace(-0.5, 0.5, 10)
    )
    
    np.random.seed(random_seed)
    # some hacky logic 
    function_chosen = divergence_free_kernel 

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


def updated_vector_field(inputs:list[np.ndarray],
                         grid_x: np.ndarray, grid_y: np.ndarray, grid_z: np.ndarray,
                         kernel_func: str, sigma_f = None, l = None, sigma_f_est = 0.1, l_est = 0.3,
                         random_seed: int = 1, 
                         noise:float = 1e-6):
    """
    Compute the GP-posterior vector field given noisy vector observations at a few points.
    returns a 3D vector field evaluated over the entire grid, conditioned on known field measurements.
    """
    sampled_points_x, sampled_points_y, sampled_points_z = inputs[0], inputs[1], inputs[2]
    sampled_fields_x, sampled_fields_y, sampled_fields_z = inputs[3], inputs[4], inputs[5]
    np.random.seed(random_seed)

    # Choose kernel

    function_chosen = divergence_free_kernel  
    
    if sigma_f is not None and l is not None: 
        print('Parameters for sigma_f and l specified, using these.')
    else:
        print('Parameters for sigma_f and l not specified, optimising these via MLE now.')
        sigma_f, l = aggregated_hyperparameter_optimisation(sampled_points_x, sampled_points_y, sampled_points_z,
                                sampled_fields_x, sampled_fields_y, sampled_fields_z, kernel_func, 
                                sigma_f_est, l_est, noise)
    
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
    # sample = np.random.multivariate_normal(posterior_mean, posterior_covar)

    # Reshape into 3D vector field
    field_vectors = posterior_mean.reshape(N, D)
    U = field_vectors[:, 0].reshape(grid_x.shape)
    V = field_vectors[:, 1].reshape(grid_y.shape)
    W = field_vectors[:, 2].reshape(grid_z.shape)

    return grid_x, grid_y, grid_z, U, V, W

