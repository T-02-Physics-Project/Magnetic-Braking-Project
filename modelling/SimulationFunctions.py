import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import random


legend = {"fontsize": 13}

# ----------------------------------------------------------------
# Functions for simulation:


def rand01():
    return random.random()


def angVelocity(time, ini_velocity, inv_tau):
    """Calculates predicted angular velocity.

    Args:
        time (np.float64): Time since electromagnet was turned on.
        ini_velocity (np.float64): Initial velocity before electromagnet is turned on.
        inv_tau (np.float64): Time constant, related to time constant due to both
                          air resistance and magnetic braking.

    Returns:
        np.float64: Angular velocity at a given time for the experimental parameters.
    """
    return np.float64(ini_velocity * np.exp(-time * inv_tau))


def invTau(tau0, m, B0):
    """Calculates (1 / time constant).

    Args:
        tau0 (np.float64): time constant due to air resistance.
        m (np.float64): constant which depends on the parameters of the experiment.
        B0 (np.float64): Magnitude of magnetic field generated by the electromagnet.

    Returns:
        np.float64: Value of (1 / time constant)
    """
    return np.float64((1 / tau0) + m * (B0 ** 2))


def critVelocity(conductivity, thickness, permeability):
    """Calculates critical velocity.

    Args:
        conductivity (float / np.float64): Electrical conductivity of disc.
        thickness (float / np.float64): Thickness of disc.
        permeability (float / np.float64): Permeability of disc.

    Returns:
        np.float64: Critical velocity.
    """
    return np.float64(2 / (conductivity * thickness * permeability))


def linToAngVelocity(linear_velocity, r):
    """Converts linear velocity to angular velocity at a given radius r on the disc.

    Args:
        linear_velocity (float / np.float64): Linear velocity of point on disc.
        r (float / np.float64): Radius of point of disc.

    Returns:
        [type]: [description]
    """
    return np.float64(linear_velocity / r)


def createPlot(xvals, yvals, label=None):
    """Creates generic plot.

    Args:
        xvals (list or list of lists): xvalues for plotting, if list of lists given attempts to plot all.
        yvals (list or list of lists): yvalues for plotting, if list of lists given attempts to plot all.
        label (string, optional): Optional label for plot. Defaults to None.

    Returns:
        [type]: [description]
    """
    legend_bool = False
    if type(xvals[0]) is list or type(xvals[0]) is np.ndarray:
        if len(xvals) > len(yvals):
            xvals = xvals[: len(yvals)]
        else:
            yvals = yvals[: len(yvals)]

        if label is not None:
            if type(label) is list:
                if len(label) == len(xvals):
                    pass
                elif len(label) > len(xvals):
                    label = label[: len(xvals)]
                elif len(label) < len(xvals):
                    label += [""] * (len(xvals) - len(label))
                legend_bool = True
        else:
            label = [""] * len(xvals)
            legend_bool = False

        colors = ['blue', 'red', 'green', 'orange', 'purple']
        if len(xvals) > len(colors):
            colors = [(rand01(), rand01(), rand01()) for val in xvals]

        fig = plt.figure(figsize=(5, 5))
        for i in range(len(xvals)):
            plt.plot(xvals[i], yvals[i],
                     "-",
                     color=colors[i],
                     linewidth=1.2,
                     label=label[i])
            plt.tick_params(direction="in", length=7)
        plt.axhline(y=0, color="black", linewidth=1.5)
        plt.axvline(x=0, color="black", linewidth=1.5)
        if legend_bool:
            plt.legend(**legend)
        plt.tight_layout()
        return fig
    else:
        if label is None:
            label = ""
        else:
            legend_bool = True
        plot = plt.plot(xvals, yvals, "-", color='blue',
                        linewidth=1, label=label)
        plt.tick_params(direction="in", length=7)
        plt.axhline(y=0, color="black", linewidth=1.2)
        plt.axvline(x=0, color="black", linewidth=1.2)
        plt.tight_layout()
        if legend_bool:
            plt.legend(**legend)
        return plot


def convToRadPerSec(rpm):
    """Converts rpm to rad/sec

    Args:
        rpm (float / np.float64): Rotations per minute

    Returns:
        (float / np.float64): Rad / sec
    """
    return (rpm / 30) * np.pi


def getExcelData(filepath, skiprows=2):
    """Returns pd.DataFrame of excel file specified

    Args:
        filepath (string): Filepath

    Returns:
        [type]: Data frame of excel data
    """
    data = pd.read_excel('improved_table.xlsx', skiprows=skiprows)
    column_headers = list(data.columns)
    return data, column_headers


def getVelocityData(data, column):
    """Retrieves data matching time & velocity data for a given column
    while filtering out NaN values.

    Args:
        data (pd.DataFrame): DataFrame data is being retrieved from.
        column (string): Column header to retrieve data from.

    Returns:
        [np.ndarray]: time values, column values
    """
    valid = ~(np.isnan(data.t) | np.isnan(data[column]))
    velocities = list(map(convToRadPerSec, data[column][valid].values))
    return data.t[valid].values, velocities


def filterHeaderArray(regex, array):
    """Filters array of header names for those matching given regex.

    Args:
        regex (byte string): Regular expression describing requirements.
        array (list, np.ndarray, etc.): Array to filter through.

    Returns:
        [list]: Filtered headers.
    """
    filter = re.compile(regex)

    return [val for val in array if filter.findall(val)]


def straight_line(x, m, c):
    return m * x + c


def getArrToFirstZero(arr):
    for i in range(len(arr)):
        if arr[i] == 0.0:
            return arr[:i]
    return arr