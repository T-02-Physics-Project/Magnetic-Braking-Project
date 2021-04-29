import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np
import os
import json
from scipy.optimize import curve_fit
from model import constant, combined, rmse, motion_dependent, mpe


def exp_(t, w_0, a, d):
    return w_0 * np.exp(-a*t) - d

def line(t, m, c):
    return m * t + c

def distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


os.chdir("C:\\Users\\ben\Desktop\\gitrepos\\physics-project\\analysis\\data")

show_plots = input("Show plotting?(y/n) ")
if show_plots == 'y':
    show_plots = True
else:
    show_plots = False

data = {}
decay_data = {}

with open("data.json", 'r') as file:
    tmp_data = json.load(file)
    data = tmp_data['low'] | tmp_data['high']
    # This is only valid in python 3.9+, comment this line & uncomment the next ones if on previous version
    # data = tmp_data['high']
    # data.update(tmp_data['low'])
with open("decay_data.json", 'r') as file:
    decay_data = json.load(file)

names = list(data.keys())

cleaned_data = {}
n = 1
for name, params in decay_data.items():
    # These are datasets this doesn't seem to work for
    invalid = []
    if name in invalid:
        continue
    time_values = data[name]['t']
    start_point = params['start']
    gap = params['gap']
    if start_point < 0:
        start_point *= -1
    time_values = [(val + start_point) for val in time_values]
    volt_values = data[name]['V']

    temp_t = 0
    colour = (np.sin(3*n)**2, np.cos(5*n)**2, abs(np.cos(5*n+3*np.pi/4)))

    if len(time_values) != len(volt_values):
        break

    w = np.array([], dtype=np.float64)
    t_w = np.array([], dtype=np.float64)
    w_error = np.array([], dtype=np.float64)

    for i in range(len(time_values)):
        time = time_values[i]
        volt = volt_values[i]
        if True:
            if volt < 20:
                if abs(volt_values[i+1] - volt) > 1:
                    if temp_t == 0:
                        temp_t = time
                    else:
                        diff = abs(time - temp_t)
                        if diff > 0.025:
                            n = w.size
                            vel = 2 * np.pi / diff
                            w_err = vel * diff / time
                            temp_t = time
                            # For first element
                            if w.size == 0:
                                w = np.insert(w, n, vel)
                                t_w = np.insert(t_w, n, time)
                                w_error = np.insert(w_error, n, w_err)
                                temp_t = time
                            # Catch extreme spikes at ends of data set, doesn't catch all so more
                            elif distance(time, vel, temp_t, w[-1]) < gap:
                                w = np.insert(w, n, vel)
                                t_w = np.insert(t_w, n, time)
                                w_error = np.insert(w_error, n, w_err)
                                temp_t = time

    # Correction for missing points etc...
    N = w.size
    i = N - 1
    while i > 0:
        time, vel = t_w[i], w[i]
        n_time, n_vel = t_w[i - 1], w[i - 1]
        if distance(time, vel, n_time, n_vel) > gap:
            if n_vel < vel:
                factor = vel / n_vel
                rounded_factor = round(factor)
                # If it can be adjusted, do
                if abs(factor - rounded_factor) < 0.1:
                    n_vel *= rounded_factor
                    w[i - 1] = n_vel
                    w_error[i - 1] *= rounded_factor
                # else remove
                else:
                    w = np.delete(w, i - 1)
                    t_w = np.delete(t_w, i - 1)
                    w_error = np.delete(w_error, i - 1)
                    N = w.size
                    i = N - 1
            elif n_vel > vel:
                w = np.delete(w, i - 1)
                t_w = np.delete(t_w, i - 1)
                w_error = np.delete(w_error, i - 1)
                N = w.size
                i = N - 1
                print("Deleting")
        i -= 1

    ang_vels = np.array([], dtype=np.float64)
    times = np.array([], dtype=np.float64)
    errors = np.array([], dtype=np.float64)
    for i in range(w.size):
        n = ang_vels.size
        valid = False
        if t_w[i] > 0:
            ang_vels = np.insert(ang_vels, n, w[i])
            times = np.insert(times, n, t_w[i])
            errors = np.insert(errors, n, w_error[i])
            valid = True
    w = ang_vels
    t_w = times
    w_error = errors

    cleaned_data.update({name: {'t_w': t_w, 'w': w, 'c': colour, 'err': w_error}})
    n += 2

mass = 0
d_mass = 7.07107e-05
thickness = 0
d_thickness = 0
moment_inertia = 0
d_moment_inertia = 0
magnetic_field = 0
d_magnetic_field = 1.73205E-05
current = 0
alpha = 1
disc_radius = 152e-3
d_disc_radius = 0.1e-3
pole_area = 1.13097e-4
d_pole_area = 2 * pole_area * 0.5e-3 / 12e-3
# mass, moment_inertia, d_moment_inertia, thickness, d_thickness, conductivity
material_data = {
    "st": (0.56975, 0.006581752, 8.16896E-07, 1.01e-3, 0.01e-3, 1.32e6),
    "cu": (0.58105, 0.00671229, 8.16897E-07, 0.9e-3, 0.01e-3, 58.7e6),
    "br": (0.53035, 0.006126603, 8.1689E-07, 0.89e-3, 0.02e-3, 15.9e6),
    "al": (0.194, 0.002241088, 8.16855E-07, 0.99, 0.01e-3, 36.9e6),
    "alth": (0.40755, 0.004739042, 8.22256e-07, 2.06e-3, 0.02e-3, 36.9e6)
}
magnetic_data = {
    "0": 0.029213333,
    "047": 0.046533333,
    "098": 0.066466667,
    "148": 0.0902076,
    "198": 0.1126426,
    "398": 0.20223826
}

plots = []
plot_names = list(cleaned_data.keys())
root_mean_square_errors = {
    "1": {
        "field1": [], "field2": [], "field3": [],
        "model1": [], "model1_name": "Combined Friction & Applied Field",
        "model2": [], "model2_name": "Motion Dependent Friction Only",
        "model3": [], "model3_name": "Constant Friction Only"
    },
    "2": {
        "field1": [], "field2": [], "field3": [],
        "model1": [], "model1_name": "Combined Friction & Applied Field",
        "model2": [], "model2_name": "Motion Dependent Friction Only",
        "model3": [], "model3_name": "Constant Friction Only"
    },
    "3": {
        "field1": [], "field2": [], "field3": [],
        "model1": [], "model1_name": "Combined Friction & Applied Field",
        "model2": [], "model2_name": "Motion Dependent Friction Only",
        "model3": [], "model3_name": "Constant Friction Only"
    }
}

mean_percentage_errors = {
    "1": {
        "field1": [], "field2": [], "field3": [],
        "model1": [], "model1_name": "Combined Friction & Applied Field",
        "model2": [], "model2_name": "Motion Dependent Friction Only",
        "model3": [], "model3_name": "Constant Friction Only"
    },
    "2": {
        "field1": [], "field2": [], "field3": [],
        "model1": [], "model1_name": "Combined Friction & Applied Field",
        "model2": [], "model2_name": "Motion Dependent Friction Only",
        "model3": [], "model3_name": "Constant Friction Only"
    },
    "3": {
        "field1": [], "field2": [], "field3": [],
        "model1": [], "model1_name": "Combined Friction & Applied Field",
        "model2": [], "model2_name": "Motion Dependent Friction Only",
        "model3": [], "model3_name": "Constant Friction Only"
    }
}

table1 = ['al047a.csv', 'al047b.csv', 'al098a.csv', 'al098b.csv', 'al148a.csv', 'al148b.csv', 'al198a.csv', 'al198b.csv', 'al398a.csv', 'al398b.csv']
table2 = ['br047a.csv', 'br047b.csv', 'br098a.csv', 'br098b.csv', 'br148a.csv', 'br148b.csv', 'br198a.csv', 'br198b.csv', 'br398a.csv', 'br398b.csv']
table3 = ['st047a.csv', 'st047b.csv', 'st098a.csv', 'st098b.csv', 'st148a.csv', 'st148b.csv', 'st198a.csv', 'st198b.csv', 'st398a.csv', 'st398b.csv']

i = 0
# Plot ln(w(t)) vs t
# Only datasets with name and start_point listed in decay_starts.csv are plotted
# Plots measurement pairs together
while i < len(plot_names):
    plot = plot_names[i][:len(plot_names[i])-4]
    low_speed = False
    high_speed = True
    material = ""

    if plot[:3] == 'low':
        low_speed = True
        high_speed = False
        # Low velocity data
        # Work out material
        material = plot[3:5]
        if plot[-2] == 'a':
            magnetic_field = magnetic_data["047"]
        else:
            magnetic_field = magnetic_data["0"]
    else:
        material = plot[:2]
        if material == 'al':                    # Aluminium has more options
            if plot[2:4] == 'th':               # Thick Al disc
                material = 'alth'
                if plot[-1] in ['a', 'b']:
                    magnetic_field = magnetic_data[plot[4:7]]
                else:
                    magnetic_field = magnetic_data["0"]
            elif plot[2:6] == 'slow':
                magnetic_field = magnetic_data["0"]
                material = 'al'
            else:                               # Thin Al disc
                material = 'al'
                if plot[-1] in ['a', 'b']:
                    magnetic_field = magnetic_data[plot[2:5]]
                else:
                    magnetic_field = magnetic_data["0"]

    mass, moment_inertia, d_moment_inertia, thickness, d_thickness, conductivity = material_data[material]
    if plot[-1] in ['a', 'b']:
        magnetic_field = magnetic_data[plot[-4:-1]]
    elif material[:2] != 'al':
        magnetic_field = magnetic_data["0"]

    plot = plot + ".csv"

    x, y = cleaned_data[plot]['t_w'], cleaned_data[plot]['w']
    c, err = cleaned_data[plot]['c'], cleaned_data[plot]['err']

    ln_y, ln_err = np.log(y), [(val1 / val2) for val1, val2 in zip(err, y)]

    # curve_fit for w data
    if high_speed:
        popt, pcov = curve_fit(exp_, x, y, sigma=err, maxfev=100000)
        perror = np.sqrt(np.diag(pcov))
        ln_popt, ln_pcov = curve_fit(line, x, ln_y, sigma=err, maxfev=100000)
        ln_perror = np.sqrt(np.diag(ln_pcov))
    else:
        popt, pcov = curve_fit(exp_, x, y, sigma=err, maxfev=100000)
        perror = np.sqrt(np.diag(pcov))
        ln_popt, ln_pcov = curve_fit(line, x, ln_y, sigma=err, maxfev=100000)
        ln_perror = np.sqrt(np.diag(ln_pcov))

    # Calculating constants
    w_0, a, d = popt[0] - popt[2], popt[1], popt[2]
    d_w_0, d_a, d_d = perror[0], perror[1], perror[2]
    w_0_e, w_0_e2 = w_0 / np.e, w_0 / (np.e ** 2)
    d_w_0_e = w_0_e * (d_w_0 / w_0)
    d_w_0_e2 = w_0_e2 * (d_w_0_e / w_0_e)
    tau, tau2 = a * np.log(w_0 / (w_0_e + d)), a * np.log(w_0_e / (w_0_e2 + d))
    d_tau = tau * np.sqrt((d_a / a)**2 + ((1 / np.log(w_0 / (w_0_e + d)))**2) * (((d_w_0_e**2 + d_d**2) / (w_0_e + d)**2) + (d_w_0 / w_0)**2))
    d_tau2 = tau2 * np.sqrt((d_a / a)**2 + ((1 / np.log(w_0_e / (w_0_e2 + d)))**2) * (((d_w_0_e2**2 + d_d**2) / (w_0_e2 + d)**2) + (d_w_0_e / w_0_e)**2))
    tau = (tau + tau2) / 2
    d_tau = np.sqrt(d_tau**2 + d_tau2**2)
    L = (4/5)*disc_radius
    d_L = L * (d_disc_radius / disc_radius)

    c_analytical = alpha * conductivity * thickness * pole_area * (L ** 2) * (moment_inertia ** 2)
    d_c_analytical = c_analytical*np.sqrt((d_thickness/thickness)**2+(2*L*d_L)**2+(2*moment_inertia*d_moment_inertia)**2)

    high_speed_zeta, low_speed_zeta = 0, 0
    if high_speed:
        high_speed_zeta = -ln_popt[0] * moment_inertia
        d_high_speed_zeta = high_speed_zeta * np.sqrt(ln_perror[0]**2 + d_moment_inertia**2)
    if low_speed:
        low_speed_zeta = moment_inertia * tau - c_analytical * (magnetic_field ** 2)
        d_low_speed_zeta = np.sqrt((moment_inertia*tau*np.sqrt((d_moment_inertia/moment_inertia)**2+(d_tau/tau)**2))**2 + (c_analytical*(magnetic_field**2)*np.sqrt((d_c_analytical/c_analytical)**2+(2*magnetic_field*d_magnetic_field/(magnetic_field**2))**2))**2)

    N = 0
    d_N = 0
    if high_speed:
        N = d*(high_speed_zeta + c_analytical * (magnetic_field**2))
        d_N = N * np.sqrt((d_d/d)**2 + (d_c_analytical/c_analytical)**2 + 4*(magnetic_field**2)*(d_magnetic_field**2))
    if low_speed:
        popt2, pcov2 = curve_fit(line, x, y, sigma=err, maxfev=30000)
        perror2 = np.sqrt(np.diag(pcov2))
        N = -popt2[0] * moment_inertia
        d_N = N * np.sqrt((perror2[0]/popt2[0])**2+(d_moment_inertia/moment_inertia)**2)
        c_alternate = ((N / d) + low_speed_zeta) / (magnetic_field ** 2)
        d_c_alternate = c_alternate * np.sqrt(((N / d)**2*((d_N/N)**2+(d_d/d)**2)+d_low_speed_zeta**2)/((N/d)+low_speed_zeta)**2 + 4*(magnetic_field**2)*(d_magnetic_field**2))

    # Produce model data
    if high_speed:
        model1 = [combined(val, w_0, N, high_speed_zeta, moment_inertia, c_analytical, magnetic_field) for val in x]
        model1_name = "Combined Friction & Applied Field"
        model2 = [motion_dependent(val, w_0, high_speed_zeta, moment_inertia) for val in x]
        model2_name = "Motion Dependent Friction Only"
        model3 = [constant(val, w_0, N, moment_inertia) for val in x]
        model3_name = "Constant Friction Only"
    else:
        model3 = [constant(val, w_0, N, moment_inertia) for val in x]
        model3_name = "Constant Friction Only"
        model1 = [combined(val, w_0, N, low_speed_zeta, moment_inertia, c_analytical, magnetic_field) for val in x]
        model1_name = "Combined Friction & Applied Field"
        model2 = [motion_dependent(val, w_0, low_speed_zeta, moment_inertia) for val in x]
        model2_name = "Motion Dependent Friction Only"

    # Evaluate model by taking Root Mean Squared Error (RMSE)
    RMSE1 = rmse(y, model1)
    RMSE2 = rmse(y, model2)
    RMSE3 = rmse(y, model3)

    MPE1 = mpe(y, model1)
    MPE2 = mpe(y, model2)
    MPE3 = mpe(y, model3)

    if plot in table1:
        root_mean_square_errors["1"]['field1'].append(magnetic_field)
        root_mean_square_errors["1"]['model1'].append(RMSE1)
        root_mean_square_errors["1"]['field2'].append(magnetic_field)
        root_mean_square_errors["1"]['model2'].append(RMSE2)
        root_mean_square_errors["1"]['field3'].append(magnetic_field)
        root_mean_square_errors["1"]['model3'].append(RMSE3)
        mean_percentage_errors["1"]['field1'].append(magnetic_field)
        mean_percentage_errors["1"]['model1'].append(MPE1)
        mean_percentage_errors["1"]['field2'].append(magnetic_field)
        mean_percentage_errors["1"]['model2'].append(MPE2)
        mean_percentage_errors["1"]['field3'].append(magnetic_field)
        mean_percentage_errors["1"]['model3'].append(MPE3)

    elif plot in table2:
        root_mean_square_errors["2"]['field1'].append(magnetic_field)
        root_mean_square_errors["2"]['model1'].append(RMSE1)
        root_mean_square_errors["2"]['field2'].append(magnetic_field)
        root_mean_square_errors["2"]['model2'].append(RMSE2)
        root_mean_square_errors["2"]['field3'].append(magnetic_field)
        root_mean_square_errors["2"]['model3'].append(RMSE3)
        mean_percentage_errors["2"]['field1'].append(magnetic_field)
        mean_percentage_errors["2"]['model1'].append(MPE1)
        mean_percentage_errors["2"]['field2'].append(magnetic_field)
        mean_percentage_errors["2"]['model2'].append(MPE2)
        mean_percentage_errors["2"]['field3'].append(magnetic_field)
        mean_percentage_errors["2"]['model3'].append(MPE3)

    elif plot in table3:
        root_mean_square_errors["3"]['field1'].append(magnetic_field)
        root_mean_square_errors["3"]['model1'].append(RMSE1)
        root_mean_square_errors["3"]['field2'].append(magnetic_field)
        root_mean_square_errors["3"]['model2'].append(RMSE2)
        root_mean_square_errors["3"]['field3'].append(magnetic_field)
        root_mean_square_errors["3"]['model3'].append(RMSE3)
        mean_percentage_errors["3"]['field1'].append(magnetic_field)
        mean_percentage_errors["3"]['model1'].append(MPE1)
        mean_percentage_errors["3"]['field2'].append(magnetic_field)
        mean_percentage_errors["3"]['model2'].append(MPE2)
        mean_percentage_errors["3"]['field3'].append(magnetic_field)
        mean_percentage_errors["3"]['model3'].append(MPE3)

    # Plotting

    # Use this variable if you want only 1 plot

    print(plot)

    plot_format = {
        "markersize": 4
    }
    label_format = {
        "fontname": "Times New Roman",
        "fontsize": 14
    }
    legend_format = {
        "prop": font_manager.FontProperties(family="Times New Roman")
    }

    fig = plt.figure(figsize=(8, 8))

    plt.errorbar(x, y, fmt="kx", yerr=err, label="Measured Data", elinewidth=0.9, capsize=2.0, **plot_format)
    if RMSE1 < 50:
        plt.plot(x, model1, "k-", linewidth=0.9, label=model1_name, **plot_format)
    if RMSE2 < 50:
        plt.plot(x, model2, "k--", linewidth=0.9, label=model2_name, **plot_format)
    if RMSE3 < 50:
        plt.plot(x, model3, "k", linestyle='dashdot', linewidth=0.9, label=model3_name, **plot_format)

    plt.xlabel("Time / s", **label_format)
    plt.ylabel(r'$\omega$(t)', **label_format)

    if max(err) > 70:
        if high_speed:
            plt.ylim(0, 250)
        else:
            plt.ylim(0, 50)

    plt.legend(**legend_format)

    if show_plots:
        plt.show()

    i += 1

with open("rmse.json", 'w') as file:
    json.dump(root_mean_square_errors, file)

with open("mpe.json", 'w') as file:
    json.dump(mean_percentage_errors, file)
