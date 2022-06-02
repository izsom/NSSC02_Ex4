import numpy as np
import matplotlib as mlp

# Group number.
groupnr = 3

# Length in x- and y-direction.
L = 0.02

# Thickness (z-direction).
hz = 0.001

# Thermal conductivity (k=k_xx=k_yy, k_xy = 0.).
k = 314.

# Factor c for modifying thermal conductivity k for
# elements in elements_to_be_modified.
c = 30.

# Elements to be modified.
elements_to_be_modified = [44-48, 60-66, 77-83, 95-99]

# Boundary conditions.
q(y=0) = 1500000.
T(y=L) = 293.
