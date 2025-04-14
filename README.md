# Repository for RT2 
Quite easily to quickly get started with this by skimming `vector_field_plottings/physical_fields.ipynb`, which contains plots of some vector fields of physical interest. Based off of these fields, in `vector_field_plottings/fitted_[EXAMPLE].ipynb` we draw samples of them and fit vector fields to them. Results are promising for divergence free kernel.  
## Repository Overview
- `GP_Implementation` Contains the scripts which allow us to make the covariance matricies from constituent 3x3 matricies. These matricies are as found in Nikolas Walstroms PhD thesis. This includes functions for: 
    1. Drawing a random sample from a kernel which is either curl or divergence free. 
    2. Updating a sample given measurements 
- `plotting_tools` Scripts which include analytical tools, including
    1. `plotter.py` Takes in vector fields and will return vector field plottings of them. 
    2. `field_analytics.py` (UNFINISHED) Takes in 2 example fields and returns multiple analytics. Including plots of residuals, and perhaps further functionality. 
- `physics_informed_vector_fields`Generates the vector fields which have physical relvance. Contains: 
    1. `constants.py` Place to store all the constants which we may make use of
    2. `current_carrying_wire.py` Field around an ideal infinite, straight, current carrying wire.
    3. `current_carrying_coil.py`Field around an ideal circular current carrying coil.
    4. `current_carrying_solenoid.py` (UNFINISHED) Field around an ideal, infinite, current carrying solenoid.
    5. `uniformly_magnetised_sphere.py` (UNFINISHED) Field around ideal, uniformly magnetised sphere. Chance to try applying boundary conditions to see if by being clever with which fields we use we can get better results. 
- `vector_field_plottings` Calls the above scripts to go over functionalities.
    1. `plotter_dev_steps.ipynb` Playing around with and testing functionality of the script at `plotting_tools/plotter.py`. Mostly Now redundant.
    2. `Physical_fields.ipynb` Examples of the vector fields which we are aiming to plot. 
    3. `kernel_samples.ipynb` Playing around with kernel parameters to invesigate sensitivity to params. 
    4. `fitted_fields.ipynb` Our updated fields fitted using GP regression.
- `tests` (EMPTY) Need to add in some of these shortly
