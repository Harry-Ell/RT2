'''
Kernel functional form  lifted from thesis by Niklas Wahlström

Attempted implementation of his kernel function. 

Each point has its own 3d vector output, at which 
you decompose into x, y and z components. Really 
unsure as to how I will go about decomosing this, 
but we will see. 

'''
import numpy as np 

def divergence_and_curl_free_kernel(x1:np.ndarray, x2:np.ndarray, sigma_f:float =1, l:float=1)->np.ndarray:
    '''
    Should be a divergence and curl free vector field which you get from here. There surely arent that 
    many of these, so this will hopefully quickly look like a magnetic field. 
    '''
    diff = x1 - x2
    # def some distance measure
    r = np.linalg.norm(diff)
    term1 = (2 - r / l) * np.eye(3)
    term2 = np.outer(diff, diff) / l**2
    scaling = np.exp(- r**2 / (2 * l**2)) * (sigma_f / l)**2
    K = (term1 + term2) * scaling
    return K


def divergence_free_kernel(x1: np.ndarray, x2: np.ndarray, sigma_f: float = 1, l: float = 1) -> np.ndarray:
    """
    Divergence-free kernel based on a squared-exponential base.
    returns a 3x3 covariance matrix for vector-valued outputs in R^3.
    """
    diff = x1 - x2
    r2 = np.dot(diff, diff)
    scaling = (sigma_f**2 / l**2) * np.exp(- r2 / (2 * l**2))
    K = scaling * (np.outer(diff, diff) / l**2 - np.eye(len(x1)))
    return K


def sample_vector_field(grid_x: np.ndarray, grid_y: np.ndarray, grid_z: np.ndarray,
                        kernel_func:str, sigma_f: float = 1, l: float = 1, random_seed: int = 1):
    """
    Given 3D meshgrid arrays (grid_x, grid_y, grid_z), build the full covariance matrix for a vector field 
    using the specified kernel function (which should return a 3x3 matrix for any two points).
    
    Returns the grid (x,y,z) and the sampled field components (U,V,W) shaped as the grid.
    
    Parameters:
      grid_x, grid_y, grid_z: 3D arrays from np.meshgrid.
      kernel_func: function with signature (x1, x2, sigma_f, l) -> (3,3) np.ndarray.
      sigma_f, l: kernel hyperparameters.
      random_seed: optional, for reproducibility.
    """

    np.random.seed(random_seed)

    points = np.vstack([grid_x.flatten(), grid_y.flatten(), grid_z.flatten()]).T
    N = points.shape[0]  # total number of spatial points
    D = 3  # dimensionality

    # Build the full covariance matrix (3N x 3N)
    cov = np.zeros((N * D, N * D))
    if kernel_func == 'divergence_free_kernel':
        for i in range(N):
            for j in range(N):
                K_ij = divergence_free_kernel(points[i], points[j], sigma_f=sigma_f, l=l)  # (3,3)
                cov[i*D:(i+1)*D, j*D:(j+1)*D] = K_ij
    else:
        for i in range(N):
            for j in range(N):
                K_ij = divergence_and_curl_free_kernel(points[i], points[j], sigma_f=sigma_f, l=l)  # (3,3)
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