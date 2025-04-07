'''    
In the case of a circulating current, a magnetic field is formed inside of it. This is a nice parity to the previous case. 
We will place our coil in the centre of the x,y plane, with current circulating anticlockwise as you look at it down the z 
axis. By the right hand rule, this will lead to a magnetic field which is aligned directly up the z axis. 
'''
from constants import CONSTANTS
import numpy as np 

def field_around_toroidal_current_carrying_wire(I:float, 
                                                ring_radius:float = 0.5,
                                                CONSTANTS:dict = CONSTANTS, 
                                                x_extent:list = [-0.5, 0.5], 
                                                y_extent:list = [-0.5, 0.5],
                                                z_extent:list = [-0.5,0.5], 
                                                resolution:float = 10, 
                                                integral_discretisation:float = 1000) -> list[np.ndarray]:
    ''' 
    Args:
        I (float): Current which the wire is carrying.
        ring_radius (float): Radius of our current carrying ring
        CONSTANTS (dict): Dictionary of all of the physical constants which could be useful.
        x_extent (list[int]): extent in the x direction of the final field. In SI Units, this will be meters.
        y_extent (list[int]): extent in the y direction of the final field. In SI Units, this will be meters.
        z_extent (list[int]): extent in the z direction of the final field. In SI Units, this will be meters.
        resolution (int): side length of rectangular grid
        integral_discretisation (int): how many segments we break up our integral into.

    Returns:
        6 numpy nd arrays 
    '''
    mu0, Pi = CONSTANTS['mu_naught'], CONSTANTS['pi']


    x, y, z = np.meshgrid(
        np.linspace(x_extent[0], x_extent[1], resolution),
        np.linspace(y_extent[0], y_extent[1], resolution),
        np.linspace(z_extent[0], z_extent[1], resolution)
    )
    Bx, By, Bz = np.zeros_like(x), np.zeros_like(y), np.zeros_like(z)


    # Biot savart prefactor
    prefactor = mu0 * I / (4*Pi)
    
    # Discretise the loop into large number of pieces for summation approximation 
    dphi = 2*np.pi / integral_discretisation
    phis = np.linspace(0, 2*Pi, integral_discretisation, endpoint=False)
    
    for phi in phis:
        # Coordinates of the source point on the loop
        xprime = ring_radius*np.cos(phi)
        yprime = ring_radius*np.sin(phi)
        zprime = 0.0
        
        dlx = -ring_radius * np.sin(phi) * dphi
        dly =  ring_radius * np.cos(phi) * dphi
        dlz =  0.0
        
        # r - r'
        rx = x - xprime
        ry = y - yprime
        rz = z - zprime
        
        # distance^3 (have to add small amount for numerical stability )
        r3 = (rx**2 + ry**2 + rz**2)**1.5 + 0.01  
        
        # cross product dl' x (r-r'):
        cross_x = dly*rz - dlz*ry  # = R cos(phi)*z * dphi
        cross_y = dlz*rx - dlx*rz  # = R sin(phi)*z * dphi
        cross_z = dlx*ry - dly*rx  
       
        # Accumulate in B
        Bx += prefactor * cross_x / r3
        By += prefactor * cross_y / r3
        Bz += prefactor * cross_z / r3

    return x, y, z, Bx, By, Bz
