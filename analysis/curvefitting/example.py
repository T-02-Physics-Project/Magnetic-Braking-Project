import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

# NOTE: I hashed this out pretty quickly, so the variable names could be better chosen.
#       It's worth spending time on naming, as it increases the readability of code considerably
#       when you know what a variable is for as soon as you read it's name.


def line(x, m, a, b):  # Define line to fit against
    return (
        m / (x + a) + b
    )  # We probably want to fit against 1/x in future, get a straight line graph.


def straight_line(x, m, c):
    return m * x + c


def convToRadPerSec(rpm):
    return ((rpm / 16) / 30) * np.pi


data = pd.read_excel("improved_table.xlsx", skiprows=2)  # Read in data
# valid = ~(np.isnan(data.t) | np.isnan(data.w))
# valid = ~(np.isnan(data))
# test = ~data.isnull()
# cleaned_data = data
# cleaned_data = data.fillna(method='ffill')                      # Clean data by forward filling all NaN values

# Get all column headers. By default, pandas.DataFrame.columns returns a pandas.Index instead of list.
column_headers = list(data.columns)


# Using regular expressions to sort out headers which relate to the angular velocity measurements.
# The 're' library is used to search for identifiers in text and filter text efficiently.
# This is a simplistic use which simple says return if the string matches the form: R[A or B or C][Any integer].
# To understand why I chose this filter, check the improved_table.xlsx file I uploaded. I've grouped all the data within and given the columns unique names.
# A, B, or C simply refers to which set of current readings it relates to.
# Look up regular expression or regex to learn more.
filter = re.compile(r"[R][A-C]\d+")

# Add all headers to list if they match the filter parameters defined above.
# We don't need time data as we just create a list of numbers with matching length to the angular velocity data
angular_velocity_headers = [
    header for header in column_headers if filter.findall(header)
]

# Concatenates an array to the first zero value. This is important because of the forward filling done before to remove NaN values.


def getArrToFirstZero(arr):
    result = []
    for el in arr:
        result.append(el)
        if el == 0.0:
            break
    return result


# Plotting isn't much different from before
# Loop iterates through the different column headers, then retrieves the relevant data from cleaned_data.
# A list of time values with matching length to the data is created starting from 0 seconds.
# Curve is fitted, very simplistic plot.
# Until we know how we're going to do our error analysis it seems excessive to put in error bars that won't exist as well.
fig = plt.figure(figsize=(20, 10))
ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)
axis = [ax1, ax2, ax3]

for ax in axis:
    ax.axhline(y=0)
    ax.axvline(x=0)

# Set up list of lists to store covariances in
stds = [[], [], []]

n = 1
for measurement in angular_velocity_headers:
    # Get current from DataFrame
    current_col = data[measurement]

    # Convert to "list" ( map* the convToRadPerSec function to all non-NaN members of the current column )
    # *map(func, **iterators) is an in-built python function which applies a function element-wise to an iterator
    # This is useful as due to being an interpretted language, python for loops are VERY slow. See the python docs for more info on map()
    angular_velocities = list(
        map(convToRadPerSec, current_col[~pd.isnull(current_col)].values)
    )
    ln_angular_velocities = np.log(
        angular_velocities[:-1]
    )  # , 0)         # Take ln() of every value except last one(=0; ln(0) -> -inf) then append last 0 back on

    # Create time values
    times = [value for value in range(len(ln_angular_velocities))]

    # Perform curve_fit
    popt, pcov = curve_fit(
        straight_line, times, ln_angular_velocities
    )  # , maxfev=100000)

    # Create theoretical values based off of curve_fit
    times2 = np.linspace(0, max(times), 1000)
    curve_fit_vals = straight_line(times2, popt[0], popt[1])

    try:
        if n < 18 or n > 25:
            raise ValueError
        m = n
        if measurement[1] == "A":
            if 5 < n < 11:
                m -= 5
            elif 10 < n < 16:
                m -= 10
            elif n > 15:
                m -= 15
            ax1.plot(
                times2, curve_fit_vals, "--", color=(0.1 * m, 1 - 0.2 * m, 0.2 * m)
            )
            ax1.plot(
                times,
                ln_angular_velocities,
                "x",
                markersize=7,
                color=(0.1 * m, 1 - 0.2 * m, 0.2 * m),
            )
            stds[0].append(np.sqrt(np.diag(pcov)))

        elif measurement[1] == "B":
            m -= 18
            ax2.plot(
                times2, curve_fit_vals, "--", color=(0.2 * m, 1 - 0.2 * m, 0.2 * m)
            )
            ax2.plot(
                times,
                ln_angular_velocities,
                "x",
                markersize=7,
                color=(0.2 * m, 1 - 0.2 * m, 0.2 * m),
            )
            stds[1].append(np.sqrt(np.diag(pcov)))

        elif measurement[1] == "C":
            m -= 24
            ax3.plot(
                times2, curve_fit_vals, "--", color=(0.2 * m, 1 - 0.2 * m, 0.2 * m)
            )
            ax3.plot(
                times,
                ln_angular_velocities,
                "x",
                markersize=7,
                color=(0.2 * m, 1 - 0.2 * m, 0.2 * m),
            )
            stds[2].append(np.sqrt(np.diag(pcov)))

    except ValueError:
        print()

    n += 1

for std in stds:
    print()
    for measurement in std:
        print(f"\t{measurement[0]}\t{measurement[1]}")

plt.show()
