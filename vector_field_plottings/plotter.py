import numpy as np
import plotly.graph_objects as go

def plotter(x, y, z, u, v, w):
    fig = go.Figure(data=go.Cone(
        x=x.flatten(), y=y.flatten(), z=z.flatten(),
        u=u.flatten(), v=v.flatten(), w=w.flatten(),
        sizemode="scaled",
        sizeref=2,  # Try adjusting this if cones are still not visible
        anchor="tail"
    ))

    fig.update_layout(scene=dict(
        xaxis_title='X', yaxis_title='Y', zaxis_title='Z'
    ))

    fig.show()
