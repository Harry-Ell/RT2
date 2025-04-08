import numpy as np 

def sample_field(inputs, n_samples: int, random_seed: int = 0):
    '''
    Helper function to use to get random points from earlier fields. Samples n points from the given field data. 

    Args: 
        inputs[0], inputs[1], inputs[2] (np.ndarray): x, y, z Meshgrid coordinates.
        inputs[3], inputs[4], inputs[5] (np.ndarray): Bx, By, Bz Magnetic field components.
        n_samples (int): Number of points to sample.

    Returns:
        list of nd.arrays of sampled x, y and zs and fields at these points
    '''
    x, y, z, Bx, By, Bz = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5]
    np.random.seed(random_seed)
    
    resolution = x.shape[0]
    indices = np.random.choice(resolution * resolution * resolution, size=n_samples, replace=False)
    
    x_samples = x.flatten()[indices]
    y_samples = y.flatten()[indices]
    z_samples = z.flatten()[indices]
    Bx_samples = Bx.flatten()[indices]
    By_samples = By.flatten()[indices]
    Bz_samples = Bz.flatten()[indices]
    
    return x_samples, y_samples, z_samples, Bx_samples, By_samples, Bz_samples