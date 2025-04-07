'''
a file here which takes in two fields, returns plots of their differences, 
and goes over some sort of mean squared error stuff perhaps and some other metrics. 

could test: average residual from fitted vs actual

other analytics also, ask henry
'''

import numpy as np 

def differencer(real_field_outputs, GPR_outputs):
    '''
    Outputs
    '''
    if any(not np.array_equal(real, gpr) for real, gpr in zip(real_field_outputs[:3], GPR_outputs[:3])):
        raise ValueError('input grids differ in dimensionality')
    res_x = GPR_outputs[3] - real_field_outputs[3]
    res_y = GPR_outputs[4] - real_field_outputs[4]
    res_z = GPR_outputs[5] - real_field_outputs[5]

    modulus = np.sqrt(res_x**2 + res_y**2 + res_z**2)

    return [GPR_outputs[0], GPR_outputs[1], GPR_outputs[2], res_x, res_y, res_z], np.mean(modulus)