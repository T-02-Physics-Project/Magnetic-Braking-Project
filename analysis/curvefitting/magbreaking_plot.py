import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

n = 1


def line(x, m, a, b):
    return m/(x + a) + b


I = 0.047
print('I={:n}A'.format(I))
while n < 14:
    data = pd.read_excel('allthedata.xlsx',
                         names=('t', 'w'),
                         usecols=(0, n))
    # | is the binary or symbol
    # a = (1100100)
    # b = (0111011)
    # c = (a | b)
    # print(c) --> (0100000)
    valid = ~(np.isnan(data.t) | np.isnan(data.w))
    popt, pcov = curve_fit(line, data.t[valid], data.w[valid], maxfev=100000)
    m = popt[0]
    a = popt[1]
    b = popt[2]
    print('for set {:n}, m={:n}, a={:n}, b={:n}' .format(n, m, a, b))

    if n < 6:

        plt.errorbar(data.t,
                     data.w,
                     xerr=0,
                     yerr=0,
                     marker='x',
                     color=(0.2*n, 1-0.2*n, 0.2*n),
                     markersize=7,
                     capsize=6,
                     linestyle='none'
                     )
        plt.plot(data.t, m/(data.t+a)+b,
                 color=(0.2*n, 1-0.2*n, 0.2*n))

    if 5 < n < 11:

        plt.errorbar(data.t,
                     data.w,
                     xerr=0,
                     yerr=0,
                     marker='x',
                     color=(1-0.2*(n-5), 0.2*(n-5), 0.2*(n-5)),
                     markersize=7,
                     capsize=6,
                     linestyle='none'
                     )
        plt.plot(data.t, m/(data.t+a)+b,
                 color=(1-0.2*(n-5), 0.2*(n-5), 0.2*(n-5)))
    if 10 < n:

        plt.errorbar(data.t,
                     data.w,
                     xerr=0,
                     yerr=0,
                     marker='x',
                     color=(0.2*(n-10), 0.2*(n-10), 1-0.2*(n-10),),
                     markersize=7,
                     capsize=6,
                     linestyle='none'
                     )
        plt.plot(data.t, m/(data.t+a)+b,
                 color=(0.2*(n-10), 0.2*(n-10), 1-0.2*(n-10),))
    n = n+1


plt.xlim(0)
plt.ylim(0)

plt.xlabel('t/s')
plt.ylabel('ω/rad$s^{-1}$')
plt.tick_params(direction='in',
                length=7
                )
plt.rcParams.update({'font.size': 19})
plt.show()

n = 16

I = 0.098
print('I={:n}A'.format(I))

while n < 22:
    data = pd.read_excel('allthedata.xlsx',
                         names=('t', 'w'),
                         usecols=(0, n))
    valid = ~(np.isnan(data.t) | np.isnan(data.w))
    popt, pcov = curve_fit(line, data.t[valid], data.w[valid], maxfev=100000)
    m = popt[0]
    a = popt[1]
    b = popt[2]
    print('for set {:n}, m={:n}, a={:n}, b={:n}' .format(n, m, a, b))

    if n-15 < 6:

        plt.errorbar(data.t,
                     data.w,
                     xerr=0,
                     yerr=0,
                     marker='x',
                     color=(0.2*(n-15), 1-0.2*(n-15), 1-0.2*(n-15)),
                     markersize=7,
                     capsize=6,
                     linestyle='none'
                     )
        plt.plot(data.t, m/(data.t+a)+b,
                 color=(0.2*(n-15), 1-0.2*(n-15), 1-0.2*(n-15)))

    if 5 < n-15 < 11:

        plt.errorbar(data.t,
                     data.w,
                     xerr=0,
                     yerr=0,
                     marker='x',
                     color=(1-0.2*(n-20), 0.2*(n-20), 0.2*(n-20)),
                     markersize=7,
                     capsize=6,
                     linestyle='none'
                     )
        plt.plot(data.t, m/(data.t+a)+b,
                 color=(1-0.2*(n-20), 0.2*(n-20), 0.2*(n-20)))
    if 10 < n-15:

        plt.errorbar(data.t,
                     data.w,
                     xerr=0,
                     yerr=0,
                     marker='x',
                     color=(0.2*(n-25), 0.2*(n-25), 1-0.2*(n-25)),
                     markersize=7,
                     capsize=6,
                     linestyle='none'
                     )
        plt.plot(data.t, m/(data.t+a)+b,
                 color=(0.2*(n-25), 0.2*(n-25), 1-0.2*(n-25)))
    n = n+1


plt.xlim(0)
plt.ylim(0)

plt.xlabel('t/s')
plt.ylabel('ω/rad$s^{-1}$')
plt.tick_params(direction='in',
                length=7
                )
plt.rcParams.update({'font.size': 19})
plt.show()

n = 24

I = 0.1475
print('I={:n}A'.format(I))

while n < 29:
    data = pd.read_excel('allthedata.xlsx',
                         names=('t', 'w'),
                         usecols=(0, n))
    valid = ~(np.isnan(data.t) | np.isnan(data.w))
    popt, pcov = curve_fit(line, data.t[valid], data.w[valid], maxfev=100000)
    m = popt[0]
    a = popt[1]
    b = popt[2]
    print('for set {:n}, m={:n}, a={:n}, b={:n}' .format(n, m, a, b))

    if n-23 < 6:

        plt.errorbar(data.t,
                     data.w,
                     xerr=0,
                     yerr=0,
                     marker='x',
                     color=(1-0.2*(n-23), 1-0.2*(n-23), 0.2*(n-23)),
                     markersize=7,
                     capsize=6,
                     linestyle='none'
                     )
        plt.plot(data.t, m/(data.t+a)+b,
                 color=(1-0.2*(n-23), 1-0.2*(n-23), 0.2*(n-23)))

    if 5 < n-23 < 11:

        plt.errorbar(data.t,
                     data.w,
                     xerr=0,
                     yerr=0,
                     marker='x',
                     color=(1-0.2*(n-28), 0.2*(n-28), 0.2*(n-28)),
                     markersize=7,
                     capsize=6,
                     linestyle='none'
                     )
        plt.plot(data.t, m/(data.t+a)+b,
                 color=(1-0.2*(n-28), 0.2*(n-28), 0.2*(n-28)))
    if 10 < n-23:

        plt.errorbar(data.t,
                     data.w,
                     xerr=0,
                     yerr=0,
                     marker='x',
                     color=(0.2*(n-33), 0.2*(n-33), 1-0.2*(n-33)),
                     markersize=7,
                     capsize=6,
                     linestyle='none'
                     )
        plt.plot(data.t, m/(data.t+a)+b,
                 color=(0.2*(n-33), 0.2*(n-33), 1-0.2*(n-33)))
    n = n+1
plt.show()

"""
plt.xlim(0)
plt.ylim(0)


plt.xlabel('t/s')
plt.ylabel('ω/rad$s^{-1}$')
plt.tick_params(direction='in',
               length=7
               )
plt.rcParams.update({'font.size':19})
plt.show()"""
