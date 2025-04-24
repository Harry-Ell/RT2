'''
Again this is mostly just for bookkeeping 

here we will have some gp regressor function which can output plots ranging from many possible curves up to 
uncertainty regions around the true values. 

this is just to be a bit more modular and also to make it clear the difference between params for input data, and also 
to make a nicer UI. 
'''

import numpy as np 
import matplotlib.pyplot as plt
from kernel_suite import Kernel

class GP_Regressor:
    '''
    Class to allow you to do gp regression
    
    intentionally sparse init method, the thinking is that this makes things a bit more malleable'''
    def __init__(self, bounds:np.array):
        self.bounds = bounds
    
    def _compute_covariance_matrix(self, 
                                   x1, 
                                   x2, 
                                   **kwargs):      
        
        '''slight generalisation of before func to allow different args'''
        n1, n2 = len(x1), len(x2)
        K = np.zeros((n1, n2))
        kernel_instance = Kernel(**kwargs)
        for i in range(n1):
            for j in range(n2):
                K[i, j] = getattr(kernel_instance, kwargs['kernel'])(x1[i], x2[j])        
        return K
    

    def priors(self, 
               kernel:str = 'rbf', 
               noise = 2,
               sigma = 1,
               var = 1/2,
               var1 = 1/2,
               var2 = 2,
               length = 0.2,
               period = 1,
               amplitude = 5,
               mean = 0,
               prior_plots:bool = False,
               prior_values:bool = False, 
               savefigs:bool = False):
        
        kernel_params = {'kernel':kernel,
                         'noise':noise, 
                         'sigma':sigma, 
                         'var':var, 
                         'var1':var1, 
                         'var2':var2, 
                         'length_scale':length, 
                         'period':period, 
                         'amp':amplitude, 
                         'mean':mean}
        '''function which will let you see where your priors are taking you after you have initialised the class'''

        assert sum([prior_plots, prior_values]) == 1, 'Only one of prior_plots and prior_values cannot be true, pick one'
        
        # define our array to populate
        xstar = list(np.linspace(self.bounds[0], self.bounds[1], 1000))
        prior_mean = np.array([mean] * len(xstar)) 
        prior_cov = np.array(self._compute_covariance_matrix(xstar, xstar, **kernel_params))


        if prior_plots == True:
            for _ in range(5):
                sample = np.random.multivariate_normal(prior_mean, prior_cov)
                plt.plot(xstar, sample, label="Sampled Function", alpha = 0.3)
            plt.title("Sample from a GP Prior")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.figure(figsize=(10,10))
            if savefigs == True:
                plt.savefig('Gp_priors_image.jpg', dpi = 400, bbox_inhces = 'tight')
            plt.show()
        elif prior_values == True:
            samples = np.random.multivariate_normal(prior_mean, prior_cov, 5)
            return {'samples':samples, 'x_values':xstar}


    def regressor(self, 
                  data_x:np.array, 
                  data_y:np.array,
                  kernel:str = 'rbf', 
                  noise = 2,
                  sigma = 1,
                  var = 1/2,
                  var1 = 1/2,
                  var2 = 2,
                  length = 0.2,
                  period = 1,
                  amplitude = 5,
                  mean = 0,
                  line_plots:bool = False, 
                  line_values:bool = False, 
                  area_plots:bool = False, 
                  area_values:bool = False,
                  savefigs:bool = False):
        '''regression function which will plot 1000 values in the bounds of the problem''' 
        assert sum([line_plots, line_values, area_plots, area_values]) == 1, 'Only one of line_plots, line_values, area_plots, area_values can be true, pick one'

        kernel_params = {'kernel':kernel,
                         'noise':noise, 
                         'sigma':sigma, 
                         'var':var, 
                         'var1':var1, 
                         'var2':var2, 
                         'length_scale':length, 
                         'period':period, 
                         'amp':amplitude}

        # points to be plotted at 
        xstar = list(np.linspace(self.bounds[0], self.bounds[1], 1000))
        prior_mean = np.array([mean] * len(xstar))

        # k matricies of use
        k_x_x = np.array(self._compute_covariance_matrix(data_x, data_x, **kernel_params)) + kernel_params['noise'] * np.identity(len(data_x))
        k_xstar_x = np.array(self._compute_covariance_matrix(xstar, data_x, **kernel_params))
        k_x_xstar = k_xstar_x.T
        k_xstar_xstar = np.array(self._compute_covariance_matrix(xstar, xstar, **kernel_params))

        # relevant variables 
        mean = np.array([mean] * len(k_xstar_xstar)) + k_xstar_x @ np.linalg.inv(k_x_x) @ (data_y - np.array([mean] * len(data_y)))
        covar = k_xstar_xstar - k_xstar_x @ np.linalg.inv(k_x_x) @ k_x_xstar
        std_dev = np.sqrt(np.diag(covar))

        # Either plot the graph, or return values to allow graph to be plotted at users discretion
        if area_plots == True:
            # Plotting the mean line
            plt.plot(xstar, mean, color="blue", label = 'Fited Mean')
            plt.scatter(data_x, data_y, color = 'black')

            # Shaded regions for various sigmas 
            plt.fill_between(xstar, mean - std_dev, mean + std_dev, color="black", alpha=0.2, label="1σ")
            plt.fill_between(xstar, mean - 2*std_dev, mean + 2*std_dev, color="black", alpha=0.15, label="2σ")
            plt.fill_between(xstar, mean - 3*std_dev, mean + 3*std_dev, color="black", alpha=0.1, label="3σ")

            # Add labels and legend
            plt.title("Mean and Confidence Intervals")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.legend()
            if savefigs == True:
                plt.savefig('Gp_CI_image.jpg', dpi = 400, bbox_inhces = 'tight')
            plt.show()
        elif line_plots == True:
            # generate 5 lines and return a plot of each of these 
            for _ in range(5):
                sample = np.random.multivariate_normal(mean, covar)
                plt.plot(xstar, sample, label="Sampled Function", alpha = 0.3)
            plt.title("Updated sample from a GP Prior")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.scatter(data_x, data_y, color = 'black')
            plt.figure(figsize=(10,10))
            if savefigs == True:
                plt.savefig('Gp_updated_priors_image.jpg', dpi = 400, bbox_inhces = 'tight')
            plt.show()
        # if we dont want plots, then give values and allow user to plot 
        
        elif line_values == True:
            samples = np.random.multivariate_normal(mean, covar, 10)
            return {'samples':samples, 'x_values':xstar, 'input_x_data': data_x, 'input_y_data':data_y}
        elif area_values == True:
            return {'x_values': xstar, 'updated_mean': mean, 'updated_std': std_dev, 'input_x_data': data_x, 'input_y_data':data_y}
        else:
            print('what on earth do you want? Choices are line_plots, line_values, area_plots, area_values, prior_plots, prior_values')
        
                   
    # a big issue with this is we would have to pass data generation function into this so it can query it itself and update 
    # its knowledge on the data

    # def optimiser(self, add_points:int = 10):
    #     '''This function will complete add_points number of bayesian optimisation steps'''
    #     xstar, mean, std_dev, _ = self.regressor(return_values=True)
    #     target_varible = mean-std_dev
    #     sampling_point = target_varible.index(min(target_varible))