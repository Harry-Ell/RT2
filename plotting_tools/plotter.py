'''
note to self: very cool functionality to add here would be a cross section 
flag: specify x, y or z and you will be returned a 2d cross section of these,
for example specifying z givves you a snapshot of the x,y plane, x gives y,z
ect.

another functionality which would be nice would be a non linear rescaling of the 
arrows to make it easier to see what is actually happenning with the fields. Softmax 
perhaps. All values at a given point need to be rescaled evenly, non equal rescalings in
x,y,z components will unfairly distort the directions of the vector fields, skewing 
the physics 
'''

import numpy as np
import plotly.graph_objects as go

def plotter(x, y, z, u, v, w, title = 'Plot of Vector Fields'):
    fig = go.Figure(data=go.Cone(
        x=x.flatten(), y=y.flatten(), z=z.flatten(),
        u=u.flatten(), v=v.flatten(), w=w.flatten(),
        sizemode="scaled",
        sizeref=2,  
        anchor="tail"
    ))

    fig.update_layout(title = title, 
        scene=dict(
        xaxis_title='X', yaxis_title='Y', zaxis_title='Z', 
    ))

    fig.show()
