'''
convenient place to collect all of my kernels 
'''
from math import sin, exp, pi

import numpy as np 

class Kernel():

    def __init__(self, **kwargs):
        self.noise = kwargs.get('noise', 1e-5)
        self.sigma = kwargs.get('sigma', 1.0)
        self.var = kwargs.get('var', 1/2)
        self.length = kwargs.get('length_scale', 0.2)
        self.period = kwargs.get('period', 1.0)  
        self.amplitude = kwargs.get('amp', 5)

    def rbf(self, x1, x2):
        '''radial basis funciton kernel'''
        return self.sigma**2 * np.exp(-0.5 * ((x1 - x2) / self.length)**2)
    
    def linear(self, x1, x2):
        '''linear kernel '''
        return x1 * x2 * self.var 
    
    def periodic(self, x1, x2):
        '''sinusoidally periodic kernel'''
        return exp(-(2/self.length**2) * (sin(pi*abs(x1 - x2) / (self.period)))**2)
    
    def quadratic(self, x1, x2):
        '''quadratic aggregate kernel'''
        return (self.linear(x1, x2)) * (self.linear(x1, x2)) * self.var
    
    def linear_plus_periodic(self, x1, x2):
        '''linear with seasonality'''
        return self.amplitude * exp(-(2/self.length**2) * (sin(pi*abs(x1 - x2) / (self.period)))**2) + (self.linear(x1, x2))