import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from scipy.optimize import curve_fit


def f(t, N, w_0, d):
    return w_0 * np.exp(-N*t) - d

os.chdir("C:\\Users\\ben\Desktop\\gitrepos\\physics-project\\analysis\\data")

data = {}
decay_starts = {}

with open("data.json", 'r') as file:
    tmp_data = json.load(file)
    data = tmp_data['high'] | tmp_data['low']       # This is only valid in python 3.9+, comment this line & uncomment the next ones if on previous version
    # data = tmp_data['high']
    # data.update(tmp_data['low'])
with open("decay_starts.csv", 'r') as file:
    reader = csv.DictReader(file, delimiter=',')
    for row in reader:
        decay_starts.update({row['n'].lower() + '.csv': np.float64(row['start'])})

cleaned_data = {}

n = 1
for name, start_point in decay_starts.items():
    print(name, start_point)
    time_values = data[name]['t']
    volt_values = data[name]['V']

    temp_t = 0
    colour = (np.sin(3*n)**2, np.cos(5*n)**2, abs(np.cos(5*n+3*np.pi/4)))

    if len(time_values) != len(volt_values):
        break

    w = np.array([], dtype=np.float64)
    t_w = np.array([], dtype=np.float64)

    for i in range(len(time_values)):
        time = time_values[i]
        volt = volt_values[i]

        if start_point < time:
            if volt < 35:
                if abs(volt_values[i+1] - volt) > 1:
                    if temp_t == 0:
                        temp_t = time
                    else:
                        if time - temp_t > 0.025:
                            n = w.size
                            vel = 2 * np.pi / (time - temp_t)
                            # For first element
                            if w.size == 0:
                                w = np.insert(w, n, 2 * np.pi / (time - temp_t))
                                t_w = np.insert(t_w, n, time)
                                temp_t = time
                            elif abs(w[-1] - vel) < 40:
                                w = np.insert(w, n, 2 * np.pi / (time - temp_t))
                                t_w = np.insert(t_w, n, time)
                                temp_t = time
                            else:
                                temp_t = time

    raw_t = np.copy(t_w)
    raw_w = np.copy(w)
    N = w.size
    x = 6

    cleaned_data.update({name: {'t_w': t_w, 'w': w, 'c': colour}})
    n += 1

start = "al01.csv"
skip = True

for name, _data in cleaned_data.items():
    if name != start and skip:
        pass
    elif not skip or name == start:
        print(name)
        skip = False
        popt, pcov = curve_fit(f, _data['t_w'], _data['w'], maxfev=30000)
        plt.figure(figsize=(10, 10))
        #plt.plot(data[name]['t'], data[name]['V'], 'r-', linewidth=0.4)
        #plt.axvline(x=decay_starts[name])
        plt.plot(_data['t_w'], _data['w'], color=_data['c'])
        #plt.plot(raw_t, raw_w, "--", linewidth=0.7)
        yvals = [f(val, popt[0], popt[1], popt[2]) for val in _data['t_w']]
        plt.plot(_data['t_w'], yvals)
        plt.show()


"""
low_data_name = str(input("Plot1? "))
high_data_name = str(input("Plot2? "))

fig = plt.figure(figsize=(20, 10))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.plot(cleaned_data[low_data_name]['t_w'], cleaned_data[low_data_name]['w'], color=cleaned_data[low_data_name]['c'])
ax2.plot(cleaned_data[high_data_name]['t_w'], cleaned_data[high_data_name]['w'], color=cleaned_data[high_data_name]['c'])

plt.show()

######

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