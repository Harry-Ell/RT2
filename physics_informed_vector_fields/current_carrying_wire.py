'''
A current carrying wire will have a magnetic field around it. The wire will run parallel to the z
axis, and be placed at the origin of the x,y plane. Current will be travelling in the direction of positive z.

Direction of the current is dependent on the direction of travel. Follows a right hand rule. 
'''

import numpy as np 
from physics_informed_vector_fields.constants import CONSTANTS

def field_around_current_carrying_wire(I:float, 
                                       CONSTANTS:dict = CONSTANTS, 
                                       x_extent:list = [-0.5, 0.5], 
                                       y_extent:list = [-0.5, 0.5],
                                       z_extent:list = [-0.5, 0.5], 
                                       resolution:float = 10) -> list[np.ndarray]:
    '''
    Args:
        I (float): Current which the wire is carrying.
        CONSTANTS (dict): Dictionary of all of the physical constants which could be useful.

        x_extent (list[int]): extent in the x direction of the final field. In SI Units, this will be meters.
        y_extent (list[int]): extent in the y direction of the final field. In SI Units, this will be meters.
        z_extent (list[int]): extent in the z direction of the final field. In SI Units, this will be meters.
        resolution (int): side length of rectangular grid
    Returns:
        6 numpy nd arrays 
    '''
    x, y, z = np.meshgrid(
        np.linspace(x_extent[0], x_extent[1], resolution),
        np.linspace(y_extent[0], y_extent[1], resolution),
        np.linspace(z_extent[0], z_extent[1], resolution)
    )

    r = np.sqrt(x**2 + y**2) + 1e-3  # avoid div by zero
    theta = np.arctan2(y, x)

    feild_strength = CONSTANTS["mu_naught"] * I / (2 * CONSTANTS["pi"])

    u = -feild_strength / r * np.sin(theta)
    v = feild_strength / r * np.cos(theta)
    w = np.zeros_like(z)

    return x, y, z, u, v, w
