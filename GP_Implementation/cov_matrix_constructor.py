'''
Here we will call the kernel we have just written and try and turn 
it into a full covariance matrix from which we can sample. At the 
moment, we will not implement any of the stochastic kriging which 
is needed to update the priors into the posteriors. Fir now this 
can just be a script which generates arbitrary covariances. 

lot more thought needed to figure out how do do this in an efficient manor. 
'''
import numpy as np 

def covariance_matrix(x:np.ndarray, y:np.ndarray, z:np.ndarray):
    '''
    the thinking here is: take the previous function which takes in 
    vector values and will give you the covariance functions.
    '''

if __name__ == "__main__":
    x, y, z =np.meshgrid(
        np.linspace(-3, 3, 10),
        np.linspace(-3, 3, 10),
        np.linspace(-3, 3, 10)
    )