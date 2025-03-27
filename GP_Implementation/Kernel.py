'''
Kernel functional form  lifted from thesis by Niklas Wahlström

Attempted implementation of his kernel function. 

Each point has its own 3d vector output, at which 
you decompose into x, y and z components. Really 
unsure as to how I will go about decomosing this, 
but we will see. 

'''
import numpy as np 
def magnetic_field_kernel(x1:np.ndarray, x2:np.ndarray, sigma_f:float =1, l:float=1)->np.ndarray:
    '''
    Should be a divergence and curl free vector field which you get from here. There surely arent that 
    many of these, so this will hopefully quickly look like a magnetic field. 
    '''
    term1 = (2 - np.abs(x1 - x2)/l) * np.eye(3)
    term2 = (x1 - x2) @ (x1-x2).T / l**2
    scaling = np.exp( - (1/(2*l**2)  * np.abs(x1 - x2)**2)) * (sigma_f/l)**2

    K_x_x = (term1 + term2) * scaling

    return np.array(K_x_x)

if __name__ == "__main__":
    x1 = np.array([3,3,1])
    x2 = np.array([1,2,0])
    cov_mat = magnetic_field_kernel(x1, x2)
    print(cov_mat)
