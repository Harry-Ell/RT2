# Repository for RT2 

## Repository Overview
- `GP_Implementation` Contains the scripts which allow us to make the covariance matricies from constituent 3x3 matricies. These matricies are as found in Nikolas Walstroms PhD thesis. This includes functions for: 
    1. Drawing a random sample from a kernel which is either curl or divergence free. 
    2. Updating a sample given measurements 
- `plotting_tools` Scripts which include analytical tools, including
    1. `plotter.py` Takes in vector fields and will return vector field plottings of them. 
    2. `field_analytics.py` Takes in 2 example fields and returns multiple analytics. Including plots of residuals, and perhaps further functionality. 
- `physics_informed_vector_fields`Generates the vector fields which have physical relvance. Contains: 
    1. `current_carrying_wire.py` Field around an ideal infinite, straight, current carrying wire.
    2. `current_carrying_coil.py`Field around an ideal circular current carrying coil.
- `vector_field_plottings` Calls the above scripts and will take in 
- `tests` Need to add in some of these shortly