import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from scipy.optimize import curve_fit


def f(t, N, w_0, d):
    return w_0 * np.exp(-N*t) - d

def g(t, m, c):
    return m * t + c

def distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

os.chdir("C:\\Users\\ben\Desktop\\gitrepos\\physics-project\\analysis\\data")

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
    invalid = []#["br01.csv", "br098a.csv", "br098b.csv", "st01.csv"]
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
            if volt < 35:
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

plot_names = list(cleaned_data.keys())
i = 0

# Plot ln(w(t)) vs t
# Only datasets with name and start_point listed in decay_starts.csv are plotted
# Plots measurement pairs together
while i < len(plot_names) - 1:
    plot1 = plot_names[i]
    plot2 = plot_names[i + 1]

    x1, y1, c1, err1 = cleaned_data[plot1]['t_w'], np.log(cleaned_data[plot1]['w']), cleaned_data[plot1]['c'], cleaned_data[plot1]['err'] / cleaned_data[plot1]['w']
    x2, y2, c2, err2 = cleaned_data[plot2]['t_w'], np.log(cleaned_data[plot2]['w']), cleaned_data[plot2]['c'], cleaned_data[plot2]['err'] / cleaned_data[plot2]['w']

    # Curve fit against motion-dependent friction only model (straight line), look at pdf I nicked from T-05
    popt1, pcov1 = curve_fit(g, x1, y1, sigma=err1, maxfev=30000)
    popt2, pcov2 = curve_fit(g, x2, y2, sigma=err2, maxfev=30000)

    plt.figure(figsize=(15, 15))

    plt.errorbar(x1, y1, yerr=err1, fmt="x", color=c1, label=plot1[:len(plot1)-4])
    plt.errorbar(x2, y2, yerr=err2, fmt="x", color=c2, label=plot2[:len(plot2)-4])

    model1 = g(x1, popt1[0], popt1[1])
    model2 = g(x2, popt2[0], popt2[1])

    plt.plot(x1, model1)
    plt.plot(x2, model2)

    plt.ylabel("ln(w(t))")
    plt.xlabel("t")

    plt.legend()

    plt.show()

    i += 2

# Uncomment to plot w(t) vs t
"""
while i < len(plot_names) - 1:
    plot1 = plot_names[i]
    plot2 = plot_names[i + 1]

    if plot1 == "br01.csv":
        print(plot1, plot2)

        x1, y1, c1, err1 = cleaned_data[plot1]['t_w'], cleaned_data[plot1]['w'], cleaned_data[plot1]['c'], cleaned_data[plot1]['err']
        x2, y2, c2, err2 = cleaned_data[plot2]['t_w'], cleaned_data[plot2]['w'], cleaned_data[plot2]['c'], cleaned_data[plot2]['err']

        plt.figure(figsize=(15, 15))

        # curve_fit can't fit to these sets for some reason
        invalid1 = ["br01.csv", "br047a.csv", "br148a.csv", "br198a.csv", "st098a.csv", "st148a.csv", "st398a.csv"]
        invalid2 = ["br02.csv", "br148b.csv", "st098b.csv", "st398b.csv"]

        plt.errorbar(x1, y1, fmt="x", yerr=None, linewidth=0.8, label=plot1)
        plt.errorbar(x2, y2, fmt="x", yerr=None, linewidth=0.8, label=plot2)

        if plot1 not in invalid1:
            popt1, pcov1 = curve_fit(f, x1, y1, sigma=cleaned_data[plot1]['err'], maxfev=30000)
            model1 = [f(val, popt1[0], popt1[1], popt1[2]) for val in x1]
            plt.plot(x1, model1, label=plot1 + " model")

        if plot2 not in invalid2:
            popt2, pcov2 = curve_fit(f, x2, y2, sigma=cleaned_data[plot2]['err'], maxfev=30000)
            model2 = [f(val, popt2[0], popt2[1], popt2[2]) for val in x2]
            plt.plot(x2, model2, label=plot2 + " model")

        plt.legend()

        plt.show()

    i += 2
"""
"""
for i in range(N - x, 0, -1):
    # print(i)
    six_values = w[i:i + x]
    rolling_avg = np.mean(six_values)
    rolling_std = np.std(six_values, ddof=1)
    deviations = [abs(val - rolling_avg) / rolling_std for val in six_values]
    for j, dev in enumerate(deviations):
        if six_values[j] > rolling_avg and dev > 3.5:  # Big big spike
            #       print("Found spike")
            w = np.delete(w, i + j)
            t_w = np.delete(t_w, i + j)
            N = w.len
            i += 1  # Recalculate 6 points
        elif six_values[j] < rolling_avg and dev > 1.2:  # Catch drops due to missing data points
            #      print("Found dip, dev = {}".format(dev))
            ratio = rolling_avg / six_values[j]
            rounded_ratio = round(ratio)
            if abs(rounded_ratio - ratio) < 0.1:  # Arbitrary difference to check for roughly integer difference
                #         print("Rounding found, ", ratio)
                w[i + j] = w[i + j] * ratio
                i += 1  # Recalculate 6 points
"""
"""else:
                                # If drop is factor
                                if vel < w[-1]:
                                    factor = w[-1] / vel
                                    rounded_factor = round(factor)
                                    if abs(factor - rounded_factor) < 0.1:
                                        print("vel = {} --> {}, w[-1] = {}".format(vel, rounded_factor * vel, w[-1]))
                                        w = np.insert(w, n, rounded_factor * vel)
                                        t_w = np.insert(t_w, n, time)
                                elif vel > w[-1]:
                                    factor = vel / w[-1]
                                    rounded_factor = round(factor)
                                    if abs(factor - rounded_factor) < 0.1:
                                        print("vel = {} --> {}, w[-1] = {}".format(vel, rounded_factor * vel, w[-1]))
                                        w[-1] = w[-1] * rounded_factor
                                        w = np.insert(w, n, vel)
                                        t_w = np.insert(t_w, n, time)"""