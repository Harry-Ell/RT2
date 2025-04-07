'''
UNFINISHED

To extend the case of a coil, we can also model an infinite current carrying solenoid, parallel to the z axis. 
This is also a case for which the analytical solution is known exactly.

Will be interesting for investigating the tail behaviour of the GPs fitted field. 
'''

import numpy as np 
from physics_informed_vector_fields.constants import CONSTANTS

def field_around_infinite_solenoid(I:float, 
                                   CONSTANTS:dict = CONSTANTS, 
                                   x_extent:list = [-5, 5], 
                                   y_extent:list = [-5, 5],
                                   z_extent:list = [5,-5], 
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
    pass