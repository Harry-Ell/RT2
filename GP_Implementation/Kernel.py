'''
Kernel functional form  lifted from thesis by Niklas Wahlström

Attempted implementation of his kernel function. 
'''

import numpy as np 
from scipy.optimize import minimize


def curl_free_kernel(x1: np.array, x2: np.array, sigma_f: float = 1, l: float = 1) -> np.ndarray:
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


def divergence_free_kernel_derivative_wrt_l(x1:np.array, x2:np.array, sigma_f:float =1, l:float=1)->np.ndarray:
    '''
    with any luck this is the derivative of the expression above wrt l. 
    '''
    diff = x1 - x2
    # def some distance measure
    r2 = np.dot(diff, diff)
    scaling = np.exp(- r2 / (2 * l**2)) * (sigma_f / l)**2
    scaling1 = (1 / l**3) * r2 - 2 /l
    scaling2 = 2/ l**3
    term1 = (2 - r2 / l**2) * np.eye(len(x1))
    term2 = np.outer(diff, diff) / l**2
    term3 = r2 * np.eye(len(x1))
    term4 = - np.outer(diff, diff)
    K = (scaling1 * (term1 + term2) + scaling2 * (term3 + term4)) * scaling
    return K 



def sample_vector_field(kernel_func:str, sigma_f: float = 1, l: float = 1, random_seed: int = 1)->np.ndarray:
    """
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
    function_chosen = divergence_free_kernel if kernel_func == 'divergence_free_kernel' else curl_free_kernel
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
    sample = np.random.multivariate_normal(posterior_mean, posterior_covar)

    # Reshape into 3D vector field
    field_vectors = sample.reshape(N, D)
    U = field_vectors[:, 0].reshape(grid_x.shape)
    V = field_vectors[:, 1].reshape(grid_y.shape)
    W = field_vectors[:, 2].reshape(grid_z.shape)

    return grid_x, grid_y, grid_z, U, V, W


def aggregated_hyperparameter_optimisation(sampled_points_x: np.ndarray, sampled_points_y: np.ndarray, sampled_points_z: np.ndarray,
                                sampled_fields_x: np.ndarray, sampled_fields_y: np.ndarray, sampled_fields_z: np.ndarray, kernel_func: str, 
                                sigma_f: float = 1, l: float = 1, noise:float = 0.1)->list[float, float]:
    '''
    hyperparameter optimisation procedure for the case of tuning both sigma_f and the length scale. 

    Perhaps also the noise/ jitter term could be incorporated into this?
    '''
    # wrapper function to call the other one. perhaps not even needed?
    if kernel_func == 'divergence_free_kernel':
        initial_guess = [sigma_f, l]
        bounds = [(0.001, None), (0.1, None)]
        # Optimize the objective
        result = minimize(
            _div_free_optimiser_wrapper, 
            initial_guess, 
            args=(sampled_points_x, sampled_points_y, sampled_points_z,
                sampled_fields_x, sampled_fields_y, sampled_fields_z, noise),
            bounds=bounds
        )
        # Extract the optimal sigma_f and l
        optimal_sigma_f, optimal_l = result.x
        print(f'Compared to initial guesses of {sigma_f} for sigma_f, and {l} for l, we have found optimums of {optimal_sigma_f} and {optimal_l}')
    else:
        print('Procedure for parameter optimisation has not been done yet for curl free kernel')
        optimal_l = l 
        optimal_sigma_f = sigma_f

    return optimal_sigma_f, optimal_l

def _div_free_optimiser_wrapper(params, spx, spy, spz, sfx, sfy, sfz, noise):
    sigma_f, l = params
    # Return negative log likelihood 
    return -_div_free_hyperparameter_value(
        sampled_points_x=spx, 
        sampled_points_y=spy, 
        sampled_points_z=spz,
        sampled_fields_x=sfx, 
        sampled_fields_y=sfy, 
        sampled_fields_z=sfz,
        sigma_f=sigma_f, 
        l=l, 
        noise=noise
    )

def _div_free_hyperparameter_value(sampled_points_x: np.ndarray, sampled_points_y: np.ndarray, sampled_points_z: np.ndarray,
                           sampled_fields_x: np.ndarray, sampled_fields_y: np.ndarray, sampled_fields_z: np.ndarray,
                           sigma_f: float = 1, l: float = 1, noise:float = 1e-3) -> float:
    '''
    Hyperparameter optimiser for the case of divergence free kernel above. This just takes in a few parameters and 
    returns a float. scipy minimize should do all the heavy lifting
    '''
    # we will start off by finding th K matrix, and the Y vector 
    points_sampled = np.vstack([sampled_points_x.flatten(), sampled_points_y.flatten(), sampled_points_z.flatten()]).T  # (M, 3)
    fields_sampled = np.vstack([sampled_fields_x.flatten(), sampled_fields_y.flatten(), sampled_fields_z.flatten()]).T  # (M, 3)    

    M = points_sampled.shape[0]  # total number of ssampled points
    D = 3  # dimensionality
    # init matricies/ vectors
    K = np.zeros((M * D, M * D))
    y = np.zeros(M * D)

    # we begin by building the covariance matrix K
    for i in range(M):
        for j in range(M):
            K_ij = divergence_free_kernel(points_sampled[i], points_sampled[j], sigma_f=sigma_f, l=l)  # (3,3)
            K[i*D:(i+1)*D, j*D:(j+1)*D] = K_ij
            
    # Build y
    for i in range(M):
        y[i*D:(i+1)*D] = fields_sampled[i]  # (3x,)?

    # finally we build the derivative of the covariance matricies
    deriv_K_sigma_f = 2 / sigma_f * K # sigma_f seems to just lead to a rescaling. 

    deriv_K_l = np.zeros((M * D, M * D))# l seems a lot more complicated
    for i in range(M):
        for j in range(M):
            K_ij = divergence_free_kernel_derivative_wrt_l(points_sampled[i], points_sampled[j], sigma_f=sigma_f, l=l)  # (3,3)
            deriv_K_l[i*D:(i+1)*D, j*D:(j+1)*D] = K_ij

    # with these, we make the trace function
    K_inv_times_y = np.linalg.solve(K + noise*np.eye(K.shape[0]), y)
    alpha = K_inv_times_y[:, np.newaxis]
    # print(y.shape, K_inv_times_y.shape, K.shape, deriv_K_l.shape)
    trace_l = 1/2 * np.trace(alpha @ alpha.T @ deriv_K_l - np.linalg.solve(K + noise*np.eye(K.shape[0]), deriv_K_l))
    trace_sigma = 1/2 * np.trace(alpha @ alpha.T @ deriv_K_sigma_f - np.linalg.solve(K + noise*np.eye(K.shape[0]), deriv_K_sigma_f))

    return trace_l + trace_sigma

def _curl_free_hyperparameter_value(sampled_points_x: np.ndarray, sampled_points_y: np.ndarray, sampled_points_z: np.ndarray,
                           sampled_fields_x: np.ndarray, sampled_fields_y: np.ndarray, sampled_fields_z: np.ndarray,
                           grid_x: np.ndarray, grid_y: np.ndarray, grid_z: np.ndarray, kernel_func: str, 
                           sigma_f: float = 1, l: float = 1, random_seed: int = 1, noise:float = 1e-3) -> float:
    '''
    Hyperparameter optimiser for the case of curl free kernel above
    '''
    pass