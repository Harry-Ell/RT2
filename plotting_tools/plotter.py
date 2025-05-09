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
import matplotlib.pyplot as plt


def plotter(inputs, title='Plot of Vector Fields', rescale = False, perspective=[1.3, 1.3, 1.3], save_as = None, write_html = False, html_name = None):
    import numpy as np
    import plotly.graph_objects as go

    # Unpack and stride
    x, y, z, u, v, w = inputs


    # Compute magnitude
    mags = np.sqrt(u**2 + v**2 + w**2)

    # Avoid division by zero
    mags = np.where(mags == 0, 1e-8, mags)
    if rescale == True:
        # Choose your nonlinear scaling
        # E.g. take square root to reduce contrast
        scale_factor = np.sqrt(mags) / mags  # Now all vectors get a shorter length
        # Or use np.log1p(mags) / mags for log-scaling

        # Rescale vectors
        u = u * scale_factor
        v = v * scale_factor
        w = w * scale_factor

    fig = go.Figure(data=go.Cone(
        x=x.flatten(),
        y=y.flatten(),
        z=z.flatten(),
        u=u.flatten(),
        v=v.flatten(),
        w=w.flatten(),
        # colorscale='Plasma',
        cmin=0,
        sizemode="scaled",
        sizeref=2,  
        anchor="tail",
        colorbar=dict(
            x=0.62,         # default is ~1.0, move left to get closer to the plot
            len=0.75,       # shrink the vertical size a bit if needed
            thickness=15,   # optional: make it slimmer
        ),
    ))

    fig.update_layout(
        title=dict(text=title,
                   x=0.5,
                   y=0.75,
                   xanchor='center',
                   font=dict(size=20)
                ),
        scene=dict(
        xaxis_title='X', yaxis_title='Y', zaxis_title='Z', 
    ))

    fig.show()


def cross_section_plotter(inputs, plane='z', title='Cross-Section of Vector Field'):
    """
    Plot a 2D cross section of a 3D vector field at the layer closest to coordinate=0 
    in the chosen plane ('x', 'y', or 'z'). The chosen plane's coordinate is ignored 
    in the 2D plot, and only the other two components are shown (as unit vectors).

    Parameters
    ----------
    x, y, z : ndarray of shape (Nx, Ny, Nz)
        Coordinate grids (e.g., created by np.meshgrid).
    u, v, w : ndarray of shape (Nx, Ny, Nz)
        Vector field components at each (x, y, z).
    plane : str, optional
        Which plane to take the cross-section of ('x', 'y', or 'z'). 
        Defaults to 'z'.
    title : str, optional
        Plot title. Defaults to 'Cross-Section of Vector Field'.
    """
    x, y, z, u, v, w = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5] 

    # Validate the plane argument
    if plane not in ('x', 'y', 'z'):
        raise ValueError("plane must be one of 'x', 'y', or 'z'.")

    # helper to pick out the slice index closest to zero
    def find_slice_index(coord_grid):
        """
        Given a grid like x[:,:,k], returns the index k 
        such that coord_grid[..., k] is closest to zero.
        """
        # We expect something like x.shape = (Nx, Ny, Nz).
        # For a plane='z', we want z[0, 0, k] to pick out the z-values along the 3rd dimension, etc.
        # Here we assume that along the dimension of interest, the other two dims can index [0].
        # The shape might differ if meshgrid was done with different indexing orders,
        # but commonly x[0,0,:] etc. should give the 1D array of that dimension.
        # If that assumption doesn't hold, you can adapt as needed.
        one_dim_array = coord_grid.take(0, axis=0).take(0, axis=0)
        # one_dim_array is now something like z[0,0,:], a 1D array along the last dimension
        idx = np.argmin(np.abs(one_dim_array))
        return idx

    # Slice and compute for each plane
    if plane == 'z':
        # Find index k closest to z=0
        k_idx = find_slice_index(z)
        
        # 2D slices
        x2d = x[:, :, k_idx]
        y2d = y[:, :, k_idx]
        
        # For vectors, ignore w and only keep (u, v)
        u2d = u[:, :, k_idx]
        v2d = v[:, :, k_idx]
        
        # Normalize each (u2d, v2d) to be unit vectors
        mag = np.sqrt(u2d**2 + v2d**2)
        # To avoid dividing by 0, add a small epsilon
        mag[mag == 0] = 1e-15
        uhat = u2d / mag
        vhat = v2d / mag
        
        # Title annotation
        slice_value = z[0, 0, k_idx]  # approximate z-value
        plot_title = f"{title}\n(Cross-section at z ≈ {slice_value:.4f})"
        
        # Make a quiver plot
        plt.figure()
        plt.quiver(x2d, y2d, uhat, vhat)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(plot_title)
        plt.axis("equal")
        plt.show()
        
    elif plane == 'x':
        # Find index i closest to x=0
        # We can use x[i,0,0] along the i dimension
        def find_x_index(xx):
            # pick out x[:, 0, 0]
            one_dim_array = xx[:, 0, 0]
            idx = np.argmin(np.abs(one_dim_array))
            return idx
        i_idx = find_x_index(x)
        
        # 2D slices in the y-z plane
        y2d = y[i_idx, :, :]
        z2d = z[i_idx, :, :]
        
        # For vectors, ignore u and only keep (v, w)
        v2d = v[i_idx, :, :]
        w2d = w[i_idx, :, :]
        
        # Normalize each (v2d, w2d)
        mag = np.sqrt(v2d**2 + w2d**2)
        mag[mag == 0] = 1e-15
        vhat = v2d / mag
        what = w2d / mag
        
        # Title
        slice_value = x[i_idx, 0, 0]
        plot_title = f"{title}\n(Cross-section at x ≈ {slice_value:.4f})"
        
        # Quiver plot
        plt.figure()
        plt.quiver(y2d, z2d, vhat, what)
        plt.xlabel("y")
        plt.ylabel("z")
        plt.title(plot_title)
        plt.axis("equal")
        plt.show()
        
    else:  # plane == 'y'
        # Find index j closest to y=0
        # We can use y[0, j, 0] along the j dimension
        def find_y_index(yy):
            one_dim_array = yy[0, :, 0]
            idx = np.argmin(np.abs(one_dim_array))
            return idx
        j_idx = find_y_index(y)
        
        # 2D slices in the x-z plane
        x2d = x[:, j_idx, :]
        z2d = z[:, j_idx, :]
        
        # For vectors, ignore v and only keep (u, w)
        u2d = u[:, j_idx, :]
        w2d = w[:, j_idx, :]
        
        # Normalize each (u2d, w2d)
        mag = np.sqrt(u2d**2 + w2d**2)
        mag[mag == 0] = 1e-15
        uhat = u2d / mag
        what = w2d / mag
        
        # Title
        slice_value = y[0, j_idx, 0]
        plot_title = f"{title}\n(Cross-section at y ≈ {slice_value:.4f})"
        
        # Quiver plot
        plt.figure()
        plt.quiver(x2d, z2d, uhat, what)
        plt.xlabel("x")
        plt.ylabel("z")
        plt.title(plot_title)
        plt.axis("equal")
        plt.show()

