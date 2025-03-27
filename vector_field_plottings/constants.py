'''
This just contains all physical constants which we will have to reuse again and again. 

Could work in h_bar = c = 1 units, but I assume the tuning of hyperparameters of the GP 
will not undo linearly in the nice way as we go to put these back in. Hnece, we will keep 
units dimensionful for now and test later if we can get away without this being the case. 
'''


from math import pi

# Universal constants
PI = pi
SPEED_OF_LIGHT = 2.99792458e8  # m/s
PLANCK_CONSTANT = 6.62607015e-34  # J·s
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K
ELEMENTARY_CHARGE = 1.602176634e-19  # C
GRAVITATIONAL_CONSTANT = 6.67430e-11  # m^3/kg/s^2
EPSILION_NAUGHT = 8.854e-12 # C^2·s^2·kg^-1·m^-3
MU_NAUGHT = 1.256e-6 # kg·m·s^-3·C^2

# Derived constants
HBAR = PLANCK_CONSTANT / (2 * pi)

# could be convenient later 
CONSTANTS = {
    "pi":PI, 
    "c": SPEED_OF_LIGHT,
    "h": PLANCK_CONSTANT,
    "hbar": HBAR,
    "k": BOLTZMANN_CONSTANT,
    "e": ELEMENTARY_CHARGE,
    "G": GRAVITATIONAL_CONSTANT,
    "epsilion_naught":EPSILION_NAUGHT,
    "mu_naught":MU_NAUGHT, 
}
