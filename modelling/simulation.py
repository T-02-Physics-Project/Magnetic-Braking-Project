import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.constants import mu_0

# User defined function imports
from SimulationFunctions import (angVelocity,
                                 invTau,
                                 critVelocity,
                                 linToAngVelocity,
                                 createPlot,
                                 getExcelData,
                                 getVelocityData,
                                 filterHeaderArray,
                                 straight_line,
                                 getArrToFirstZero)

# ----------------------------------------------------------------
# Experiment parameters:
# Some values taken from "American Journal of Physics 55, 500 (1987); doi: 10.1119/1.15103" when
# no respective value was provided in data. These values are marked *

permeability = mu_0
conductivity = 2.4187562189e7  # *Electrical conductivity of disc. [(ohm m)^-1]
disc_radius = 0.3                       # Radius of disc. [m]
thickness = 2e-3                        # Thickness of disc. [m]
air_gap = 4e-3  # *Size of air gap. [m]
w = 29e-3  # *Width of strip.  [m]
l = 57e-3  # *Length of strip.  [m]
L = 0.2  # *Dist. from axis of rotation to centre of strip.
# *External resistance of strip (area outside shadow of the magnet poles).
R = 0.87 / (conductivity * thickness)
r = w / (conductivity * l * thickness)  # *Internal resistance of strip.
K = 1.18e-2  # *Moment of inertia of disc.  [kg^-2]
tau0 = 1  # *Time constant due to air resistance.  [s^-1]
# Magnitude of magnetic field produced by electromagnet.  [T]
B0 = 55.3e-3
m = (conductivity * thickness * l * w * (L**2)) / ((1 + (R / r)) * K)
# ^ constant used to calculated time constant due to magnetic braking.
W0 = 50      # Initial angular velocity.  [rad s^-1]
W_critical = linToAngVelocity(critVelocity(
    conductivity, thickness, permeability), L)

axis_label = {"fontname": 'Times New Roman', "fontsize": 20}

# ----------------------------------------------------------------
# Beginning of simulation:

# ----------------------------------
# Simulations for varying thickness:

# Setting up variables beforehand
thicknesses = [1, 0.1, 0.01, 0.001, 0.0001]
time = np.linspace(0.00001, 5, 10000)
times = []
ln_ang_vels = []
labels = []

# Recalculating dependent variables for different thicknesses.
for t in thicknesses:
    R = 0.87 / (conductivity * thickness)
    r = w / (conductivity * l * thickness)
    m = (conductivity * t * l * w * (L**2)) / ((1 + (R / r)) * K)

    v = np.log(angVelocity(time, W0, invTau(tau0, m, B0)))
    bools = v > 0
    v = v[bools]

    ln_ang_vels.append(v)
    times.append(time[:len(v)])
    labels.append("{:.2e} / m".format(t))

# Creating the plots.
createPlot(times, ln_ang_vels, labels)

# Plotting
plt.xlabel("time / s", **axis_label)
plt.ylabel(r"Ln($\omega$)", **axis_label)
plt.show()

# -------------------------------------
# Simulations for varying conductivity:

conductivities = [1e6, 1e7, 2.4187562189e7, 1e8]
times = []
ln_ang_vels = []
labels = []

for c in conductivities:
    R = 0.87 / (c * thickness)
    r = w / (c * l * thickness)
    m = (c * thickness * l * w * (L**2)) / ((1 + (R / r)) * K)

    v = np.log(angVelocity(time, W0, invTau(tau0, m, B0)))
    bools = v > 0
    v = v[bools]

    ln_ang_vels.append(v)
    times.append(time[:len(v)])
    labels.append("{:.2e}".format(c) + r" / $\Omega^{-1}$$m^{-1}$")

# Creating plots
createPlot(times, ln_ang_vels, labels)

# Plotting
#plt.xlim(0.3, 0.4)
#plt.ylim(0, 1)
plt.xlabel("time / s", **axis_label)
plt.ylabel(r"Ln($\omega$)", **axis_label)
plt.show()

# ----------------------------------------------
# Simulations for time const. dependence on B^2:

flux = np.linspace(0, 100e-3, 10000)
inv_taus = []
labels = []

for f in flux:
    inv_taus.append(invTau(tau0, m, f))

createPlot(flux ** 2, inv_taus)

plt.xlabel(r"$B^{2}$ / $T^{2}$", **axis_label)
plt.ylabel(r"$\tau^{-1}$ / $s^{-1}$", **axis_label)

plt.show()

# ----------------------------------------------
# Checking measured inverse tau predictions

data, headers = getExcelData('improved_table.xlsx')
ang_headers = filterHeaderArray(r"[R][A-C]\d+", headers)

popts = {38.6: [], 55.3: [], 72.4: []}
pcovs = {38.6: [], 55.3: [], 72.4: []}
perrors = {38.6: [], 55.3: [], 72.4: []}
taus = {38.6: [], 55.3: [], 72.4: []}

for header in ang_headers:
    time, vel = getVelocityData(data, header)
    vel = getArrToFirstZero(vel)
    time = time[:len(vel)]
    vel = np.log(vel)
    popt, pcov = curve_fit(straight_line, time, vel)

    if header[1] == 'A':
        popts[38.6].append(popt)
        pcovs[38.6].append(pcov)
        perrors[38.6].append(np.sqrt(np.diag(pcov)))
        taus[38.6].append(-popt[0])
    elif header[1] == 'B':
        popts[55.3].append(popt)
        pcovs[55.3].append(pcov)
        perrors[55.3].append(np.sqrt(np.diag(pcov)))
        taus[55.3].append(-popt[0])
    elif header[1] == 'C':
        popts[72.4].append(popt)
        pcovs[72.4].append(pcov)
        perrors[72.4].append(np.sqrt(np.diag(pcov)))
        taus[72.4].append(-popt[0])

low_error_taus = {38.6: [], 55.3: [], 72.4: []}
std_devs = {38.6: [], 55.3: [], 72.4: []}

for ((k1, params), errors) in zip(taus.items(), perrors.values()):
    for param, error in zip(params, errors):
        if (error[0] / param) < 0.15:
            low_error_taus[k1].append(param)
            std_devs[k1].append(error[0])
            print(f"{k1}: {param} +- {error[0]}")

