'''
This would be the classical bar magnet field. probably the one most people have seen. 

This is very closely related to the current carrying coil, and formally all we have 
done is tend the radius of the ring to 0 (i think). 
'''

import numpy as np 
from physics_informed_vector_fields.constants import CONSTANTS


def field_around_magnetic_diplole(m:float, 
                                  CONSTANTS:dict = CONSTANTS, 
                                  x_extent:list = [-0.5, 0.5], 
                                  y_extent:list = [-0.5, 0.5],
                                  z_extent:list = [-0.5, 0.5], 
                                  resolution:float = 10, 
                                  masking:float = 0.05) -> list[np.ndarray]:
    ''' 
    We will be assuming this behaves like an infinitessimal point at the origin. 
    Args:
        m (float): Magnitude of magnetic Dipole.
        CONSTANTS (dict): Dictionary of all of the physical constants which could be useful.
        x_extent (list[int]): extent in the x direction of the final field. In SI Units, this will be meters.
        y_extent (list[int]): extent in the y direction of the final field. In SI Units, this will be meters.
        z_extent (list[int]): extent in the z direction of the final field. In SI Units, this will be meters.
        resolution (int): side length of rectangular grid
        masking (float): region near singularity where we set vector field to 0. 
    Returns:
        6 numpy nd arrays 
    '''
    x, y, z = np.meshgrid(
        np.linspace(x_extent[0], x_extent[1], resolution),
        np.linspace(y_extent[0], y_extent[1], resolution),
        np.linspace(z_extent[0], z_extent[1], resolution),
        indexing = 'ij')

    r = np.sqrt(x**2 + y**2 + z**2) + 1e-8  # avoid div by zero

    mu0, Pi = CONSTANTS['mu_naught'], CONSTANTS['pi']
    prefactor = mu0 / (4*Pi)

    
    term1 = 3 * m*z / r**5
    term2 = m / r**3

    u = prefactor * (x*term1)
    v = prefactor * (y*term1)
    w = prefactor * (z*term1 - term2)
    # to make it so we can actually see things on the plots
    masking = 0.05 
    mask = r < masking

    
    u[mask] = 0
    v[mask] = 0
    w[mask] = 0

    return x, y, z, u, v, w
