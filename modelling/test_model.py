import numpy as np
import matplotlib.pyplot as plt


# All lengths in m, areas in m^2, unless otherwise noted a list indicates multiple values were used
disc_rad = 190e-3
air_gap = [1e-3, 3e-3, 5e-3]
n_turns = 250
iron_core_diameter = 60e-3
disc_thickness = [4e-3, 5e-3]
pole_diameter = 60e-3
disc_pole_distance = 70e-3
pole_area = 2.828e-3
air_permeability = 12.568e-7                    # N.A^{-2}
# Conductivity of Al6061 & Al7075 respectively | [Ohm / m]
electrical_conductivity = [2.73e7, 1.92e7]


def flux_density(current, air_gap, n_turns=n_turns, mu_0=air_permeability):
    return np.float64((mu_0 * n_turns * current) / air_gap)


def braking_force(current, air_gap, disc_thickness, electrical_conductivity, angular_velocity,
                  n_turns=n_turns,
                  mu_0=air_permeability,
                  disc_pole_distance=disc_pole_distance,
                  pole_area=pole_area):
    flux = flux_density(current, air_gap, n_turns, mu_0)
    return electrical_conductivity * (disc_pole_distance ** 2) * pole_area * disc_thickness * angular_velocity * (flux ** 2) * (current ** 2)


fig = plt.figure(figsize=(10, 10))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

current = 10
ang_velocities = np.linspace(0.0, 2000, 1000000)

for thickness, conductivity in zip(disc_thickness, electrical_conductivity):
    torque_1mm = braking_force(
        current, air_gap[0], thickness, conductivity, ang_velocities)
    torque_3mm = braking_force(
        current, air_gap[1], thickness, conductivity, ang_velocities)
    torque_5mm = braking_force(
        current, air_gap[2], thickness, conductivity, ang_velocities)

    ax1.plot(ang_velocities, torque_1mm, label=f"thickness={thickness}")
    ax2.plot(ang_velocities, torque_3mm, label=f"thickness={thickness}")
    ax3.plot(ang_velocities, torque_5mm, label=f"thickness={thickness}")

ax1.legend()
ax2.legend()
ax3.legend()

plt.tight_layout()
plt.show()
