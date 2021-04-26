import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Constants

print("Please provide values in SI units, eg kg, m, s.\n")

mass = np.float64(input("What is the mass of the disc? "))                                  # mass of disc
radius = np.float(input("What is the radius of the disc? "))                                # radius of disc
disc_area = np.pi * (radius ** 2)                                                           # face area of disc
mom_inertia = 0.5 * mass * (radius ** 2)                                                    # moment of inertia of disc
B = np.float64(input("What is the applied magnetic field? "))                               # applied magnetic field strength
omega_0 = np.float64(input("What is the initial angular velocity? "))                       # initial angular velocity
conductivity = np.float64(input("What is the conductivity of the disc? "))                  # conductivity of disk
thickness = np.float64(input("What is the thickness of the disc? "))                        # thickness of disc
width = np.float64(input("What is the width of area under field? "))                        #
length = np.float64(input("What is the length of area under under field? "))                #
field_area = width * length                                                                 #
L = np.float64(input("What is the distance from centre of disc to centre of magnets? "))    # dist from centre of disc --> centre of magnetic field area
tau_0 = np.float64(input("What is the time constant, tau_0? "))                             # this value is calculated from the data, time for value to fall to 1/e initial value
internal_resistance = width / (conductivity * length * thickness)                           #
external_resistance = (1 / conductivity) * (2 * radius) / disc_area                         #
n = np.float64(input("What is the measured value of N? "))                                  # measured from gradient of low speed regime
t_endpoint = np.float64(input("What is the endpoint of the time? "))                        #
n_values = int(input("How many values? "))                                                  #


alpha = 1 / (1 + (internal_resistance / external_resistance))                               #
zeta = mom_inertia / tau_0                                                                  #
c = alpha * conductivity * thickness * field_area * (L ** 2) * (mom_inertia ** 2)           #
d = n / (zeta + c * (B ** 2))                                                               #
omega_prime = omega_0 + d                                                                   #


# Functions


def omega_t(t):
    return omega_prime * np.exp(-(zeta + c * (B ** 2) * t) / mom_inertia) - d


time_values = np.linspace(0, t_endpoint, n_values)
velocity_values = omega_t(time_values)

plt.plot(time_values, velocity_values)
plt.show()
