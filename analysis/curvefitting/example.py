import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

# NOTE: I hashed this out pretty quickly, so the variable names could be better chosen.
#       It's worth spending time on naming, as it increases the readability of code considerably
#       when you know what a variable is for as soon as you read it's name.

def line(x, m, a, b):                                           # Define line to fit against
    return m / (x + a) + b                                      # We probably want to fit against 1/x in future, get a straight line graph.

data = pd.read_excel('improved_table.xlsx', skiprows=2)         # Read in data
cleaned_data = data.fillna(method='ffill')                      # Clean data by forward filling all NaN values

column_headers = list(cleaned_data.columns)                     # Get all column headers. By default, pandas.DataFrame.columns returns a pandas.Index instead of list.

# Using regular expressions to sort out headers which relate to the angular velocity measurements.
# The 're' library is used to search for identifiers in text and filter text efficiently.
# This is a simplistic use which simple says return if the string matches the form: R[A or B or C][Any integer].
# To understand why I chose this filter, check the improved_table.xlsx file I uploaded. I've grouped all the data within and given the columns unique names.
# A, B, or C simply refers to which set of current readings it relates to.
# Look up regular expression or regex to learn more.
filter = re.compile(r'[R][A-B]\d+')

# Add all headers to list if they match the filter parameters defined above.
# We don't need time data as we just create a list of numbers with matching length to the angular velocity data
angular_velocity_headers = [header for header in column_headers if filter.findall(header)]

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
for measurement in angular_velocity_headers:
    angular_velocities = getArrToFirstZero(cleaned_data[measurement].values)
    times = [value for value in range(len(angular_velocities))]

    popt, pcov = curve_fit(line, times, angular_velocities, maxfev=100000)

    times2 = np.linspace(0, max(times)+1, 1000)
    curve_fit_vals = line(times2, popt[0], popt[1], popt[2])

    plt.plot(times2, curve_fit_vals, "r--")
    plt.plot(times, angular_velocities, "kx")
    plt.show()

