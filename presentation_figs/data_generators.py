'''
script where i will place all the random data generation processes I may try and test on

literally just to make code a bit more readable and modular, ability to import data

in final project i will probably just use actual data from some real set 
'''
import numpy as np 
from math import sin

class Data:
    def __init__(self, 
                 number_of_points:int, 
                 bounds:list[int, int]= [0,100]):
        
        self.n = number_of_points
        self.bounds = bounds

    def _rescaling(self, x):
        return self.bounds[0] + (x/100)*(self.bounds[1] - self.bounds[0])

    def linear(self, noise:float = 0, gradient:float = 0):
        true_data = np.array([i*gradient for i in range(100)])
        sample_indices = np.random.choice(true_data.shape[0], size=self.n, replace=False)
        x_values_sampled_data = self._rescaling(sample_indices)
        x_values_all_data = self._rescaling(np.linspace(0,100,100))
        return [true_data[sample_indices][i] + np.random.normal(0, noise) for i in range(self.n)], x_values_sampled_data, true_data, x_values_all_data
    
    def quadratic(self, noise:float = 0, gradient:float = 0):
        true_data = np.array([gradient *(i)**2 for i in range(100)])
        sample_indices = np.random.choice(true_data.shape[0], size=self.n, replace=False)
        x_values_sampled_data = self._rescaling(sample_indices)
        x_values_all_data = self._rescaling(np.linspace(0,100,100))
        return [true_data[sample_indices][i] + np.random.normal(0, noise) for i in range(self.n)], x_values_sampled_data, true_data, x_values_all_data
    
    def sinusoid(self, noise:float = 0, period:float = 2, amplitude:float = 2):
        true_data = np.array([sin(i / period) for i in range(100)]) * amplitude
        sample_indices = np.random.choice(true_data.shape[0], size=self.n, replace=False)
        x_values_sampled_data = self._rescaling(sample_indices)
        x_values_all_data = self._rescaling(np.linspace(0,100,100))
        return [true_data[sample_indices][i] + np.random.normal(0, noise) for i in range(self.n)], x_values_sampled_data, true_data, x_values_all_data
    
    def quadratic_sinusoid(self, noise:float = 0, gradient:float = 0, period:float = 2):
        true_data = np.array([sin(i*period) + gradient * i **2 for i in range(100)])
        sample_indices = np.random.choice(true_data.shape[0], size=self.n, replace=False)
        x_values_sampled_data = self._rescaling(sample_indices)
        x_values_all_data = self._rescaling(np.linspace(0,100,100))
        return [true_data[sample_indices][i] + np.random.normal(0, noise) for i in range(self.n)], x_values_sampled_data, true_data, x_values_all_data
    
    def linear_sinusoid(self, noise:float = 0, gradient:float = 0, period:float = 2, amplitude:float = 2):
        true_data = np.array([amplitude * sin(i*period) + gradient * i  for i in range(100)])
        sample_indices = np.random.choice(true_data.shape[0], size=self.n, replace=False)
        x_values_sampled_data = self._rescaling(sample_indices)
        x_values_all_data = self._rescaling(np.linspace(0,100,100))
        return [true_data[sample_indices][i] + np.random.normal(0, noise) for i in range(self.n)], x_values_sampled_data, true_data, x_values_all_data
 
    