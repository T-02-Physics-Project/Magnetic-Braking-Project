import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
#Most of this code is just re-used/ straight up butchered from Ben's stuff, so if it looks well refined I probably didn't write it

def line(x, m, c):
    return m*x+c
def line2(x,a,b):
    return np.exp(x*a+b)
#Defines the lines that the curve_fit will fit to- line2 function is defined according to log rules (I might have to tighten these up though)

grad=[0]
intercept=[0]
grad2=[0]
intercept2=[0]
a1=[0]
a2=[0]
b1=[0]
b2=[0]
v=[0]
AM1=[0]
AM2=[0]
AM3=[0]
#Defines the arrays that will store the data from the log graph       

data = pd.read_excel('improved_table_2.xlsx', skiprows=2)         # Read in the natural log of the data
column_headers = list(data.columns)                     # Get all column headers. By default, pandas.DataFrame.columns returns a pandas.Index instead of list.

# Using regular expressions to sort out headers which relate to the angular velocity measurements.
# The 're' library is used to search for identifiers in text and filter text efficiently.
# This is a simplistic use which simple says return if the string matches the form: R[A or B or C][Any integer].
# To understand why I chose this filter, check the improved_table.xlsx file I uploaded. I've grouped all the data within and given the columns unique names.
# A, B, or C simply refers to which set of current readings it relates to.
# Look up regular expression or regex to learn more.
filter = re.compile(r'[R][A-C]\d+')

# Add all headers to list if they match the filter parameters defined above.
angular_velocity_headers = [header for header in column_headers if filter.findall(header)]

fig = plt.figure(figsize=(50, 20))
ax1 = fig.add_subplot(251)
ax2 = fig.add_subplot(252)
ax3 = fig.add_subplot(253)
ax4 = fig.add_subplot(256)
ax5 = fig.add_subplot(257)
ax6 = fig.add_subplot(258)
ax10 = fig.add_subplot(2,5,5)
ax11 = fig.add_subplot(2,5,10)
#Defines how each plot shall be shown in one figure- I guess it can be cropped if it doesn't fit

ax1.set_xlabel('t/s')
ax1.set_ylabel('ln(ω/rad$s^{-1}$)')
ax2.set_xlabel('t/s')
ax2.set_ylabel('ln(ω/rad$s^{-1}$)')
ax3.set_xlabel('t/s')
ax3.set_ylabel('ln(ω/rad$s^{-1}$)')
ax4.set_xlabel('t/s')
ax4.set_ylabel('ω/rad$s^{-1}$')
ax5.set_xlabel('t/s')
ax5.set_ylabel('ω/rad$s^{-1}$')
ax6.set_xlabel('t/s')
ax6.set_ylabel('ω/rad$s^{-1}$')
ax10.set_xlabel('t/s')
ax10.set_ylabel('ω/rad$s^{-1}$')
ax11.set_xlabel('t/s')
ax11.set_ylabel('ω/rad$s^{-1}$')
#Labels for each axis


n = 1
#Used to count each column of data
for measurement in angular_velocity_headers:
    angular_velocities = data[measurement].values
    #Allows us to repeatedly calculate curve_fits for each seperate set of measurements
    
    c=(np.sin(n/2)**2,np.cos(n*3/4)**2, abs(np.cos(n*5+np.pi/4)))
    #As colour takes 3 arguments between 0 and 1, I used variations on sin and cos functions to make colours vary, and to allow the same colour to be used for the data and the line of best fit
    
    valid = (np.isfinite(angular_velocities))
    lndata=np.log(angular_velocities)
    #This flags A LOT of errors- this is because is taking the log of the NaNs and zeroes- there is probably a workaround but it works for what we need
    valid1 = (np.isfinite(lndata))

    x=lndata>2
    y=lndata<2.5
    valid2 = valid1*x
    valid3 = valid1*y
    #This creates an array of 0s and 1s to test whether or not there is data present, and then only allows points were there is actual data to be plotted, to avoid a NaN error
    #It also allows us to test if the data follows the first or second curve
    
    v.append(valid3)
    popt1, pcov1 = curve_fit(line2, data.t[valid], angular_velocities[valid], maxfev=1000000)

    if sum(valid1)>2:
        popt, pcov = curve_fit(line, data.t[valid2], lndata[valid2], maxfev=1000000)
        if sum(valid3)<2:
            times2 = np.linspace(0,max(data.t),1000)
            lncurve_fit_vals2 = line(times2, 0, -5)
            times3= np.linspace(0,1,1000)
            popt3=[-10,-10]
            popt2=[-10,-10]
        else:
            popt2, pcov2 = curve_fit(line, data.t[valid3], lndata[valid3], maxfev=1000000)
            #curve_fit for the ln function
            times2 = np.linspace(0, max(data.t[valid3]), 1000)
            times3 = np.linspace(min(data.t[valid3]), max(data.t), 1000)
            #defines an array to use to plot a line of best fit
            lncurve_fit_vals2 = line(times3, popt2[0], popt2[1])
            popt3, pcov3 = curve_fit(line2, data.t[valid3], angular_velocities[valid3], maxfev=100000)
        
        grad.append(popt[0])
        intercept.append(popt[1])
        grad2.append(popt2[0])
        intercept2.append(popt2[1])
        a1.append(popt1[0])
        b1.append(popt1[1])
        a2.append(popt3[0])
        b2.append(popt3[1])
        #these arrays are used to record the values for the fits, so they can be printed at the end of the program.
    else:
        popt, pcov = curve_fit(line, data.t[valid1], lndata[valid1], maxfev=1000000)
        times2 = np.linspace(0,max(data.t),1000)
        times3= np.linspace(0,1,1000)
        popt3=[-10,-10]
        lncurve_fit_vals2 = line(times2, 0, -5)
        grad.append(popt[0])
        intercept.append(popt[1])
        grad2.append(-10)
        intercept2.append(-10)
        a1.append(popt1[0])
        b1.append(popt1[1])
        a2.append(-10)
        b2.append(-10)
    #this if clause exists so that there arent errors created by trying to plot RA0, as it doesnt fit the criteria for valid2 or valid3, which exist to separate where the data splits from one exponential to the other
    lncurve_fit_vals = line(times2, popt[0], popt[1])
    curve_fit_vals = line2(times2, popt1[0], popt1[1])
    curve_fit_vals2 = line2(times3, popt3[0], popt3[1])
    #creates a set of data to be plotted agaisnt times2

    
    #these append the previously defined arrays so they can be used in equation line2        

    if measurement[1] == "A":
        #tests which set of data point "n" falls in, so it can separate different data by the presence of magnetic fields
        ax1.plot(times2, lncurve_fit_vals, 
                 color=(c),
                 linestyle="-")
        ax1.plot(times3, lncurve_fit_vals2, 
                 color=(c),
                 linestyle="-")        
        ax1.plot(data.t,
                 lndata,
                 color=(c),
                 marker="x",
                 markersize=7,
                 linestyle='none'
                 )
        AM1.append(popt1[0])
        #plots the line of best fit        
        if grad2[n]==-10:
            ax4.plot(times2, curve_fit_vals, 
                     color=(c),
                     linestyle="-")
            ax4.plot(data.t,
                     angular_velocities,
                     color=(c),
                     marker="x",
                     markersize=7,
                     linestyle='none'
                     )    
        else:
            ax4.plot(times2, curve_fit_vals, 
                     color=(c),
                     linestyle="-")
            ax4.plot(times3, curve_fit_vals2, 
                     color=(c),
                     linestyle="-")        
            ax4.plot(data.t,
                     angular_velocities,
                     color=(c),
                     marker="x",
                     markersize=7,
                     linestyle='none'
                     )    
        #plots the data with a newly calculated fit
        #these steps are the same for B and C
    elif measurement[1] == "B":
        ax2.plot(times2, lncurve_fit_vals, 
                 color=(c),
                 linestyle="-")
        ax2.plot(times3, lncurve_fit_vals2, 
                 color=(c),
                 linestyle="-")
        ax2.plot(data.t,
                 lndata,
                 color=(c),
                 marker="x",
                 markersize=7,
                 linestyle='none'
                 )
        if grad2[n]==-10:
            ax5.plot(times2, curve_fit_vals, 
                     color=(c),
                     linestyle="-")
            ax5.plot(data.t,
                     angular_velocities,
                     color=(c),
                     marker="x",
                     markersize=7,
                     linestyle='none'
                     )    
        else:
            ax5.plot(times2, curve_fit_vals, 
                     color=(c),
                     linestyle="-")
            ax5.plot(times3, curve_fit_vals2, 
                     color=(c),
                     linestyle="-")        
            ax5.plot(data.t,
                     angular_velocities,
                     color=(c),
                     marker="x",
                     markersize=7,
                     linestyle='none'
                     ) 
        AM2.append(popt1[0])
    elif measurement[1] == "C":
        ax3.plot(times2, lncurve_fit_vals, 
                 color=(c),
                 linestyle="-")
        ax3.plot(times3, lncurve_fit_vals2, 
                 color=(c),
                 linestyle="-")
        ax3.plot(data.t,
                 lndata,
                 color=(c),
                 marker="x",
                 markersize=7,
                 linestyle='none'
                 )
        if grad2[n]==-10:
            ax6.plot(times2, curve_fit_vals, 
                     color=(c),
                     linestyle="-")
            ax6.plot(data.t,
                     angular_velocities,
                     color=(c),
                     marker="x",
                     markersize=7,
                     linestyle='none'
                     )    
        else:
            ax6.plot(times2, curve_fit_vals, 
                     color=(c),
                     linestyle="-")
            ax6.plot(times3, curve_fit_vals2, 
                     color=(c),
                     linestyle="-")        
            ax6.plot(data.t,
                     angular_velocities,
                     color=(c),
                     marker="x",
                     markersize=7,
                     linestyle='none'
                     ) 
        AM3.append(popt1[0])
    #prints all the data on one plot, instead of separating by magnetic field- in case this becomes relevant
    ax10.plot(times2, lncurve_fit_vals, 
              color=(c),
              linestyle="-")
    ax10.plot(times3, lncurve_fit_vals2, 
             color=(c),
             linestyle="-")
    ax10.plot(data.t,
              lndata,
              color=(c),
              marker="x",
              markersize=7,
              linestyle='none'
              )
    if grad2[n]==-10:
        ax11.plot(times2, curve_fit_vals, 
                  color=(c),
                  linestyle="-")
        ax11.plot(data.t,
                   angular_velocities,
                   color=(c),
                   marker="x",
                   markersize=7,
                   linestyle='none',
                   label=measurement
                   )
    else:
         ax11.plot(times2, curve_fit_vals, 
                  color=(c),
                  linestyle="-")
         ax11.plot(times3, curve_fit_vals2, 
                   color=(c),
                   linestyle="-")    
         ax11.plot(data.t,
                   angular_velocities,
                   color=(c),
                   marker="x",
                   markersize=7,
                   linestyle='none',
                   label=measurement
                   )
    
    n += 1
    #moves on to the next measurement

ax1.set_xlim(0,22)
ax1.set_ylim(-0.5,4.2)
ax2.set_xlim(0,22)
ax2.set_ylim(-0.5,4.2)
ax3.set_xlim(0,22)
ax3.set_ylim(-0.5,4.2)
ax4.set_xlim(0,22)
ax4.set_ylim(0)
ax5.set_xlim(0,22)
ax5.set_ylim(0)
ax6.set_xlim(0,22)
ax6.set_ylim(0)
ax10.set_xlim(0,22)
ax10.set_ylim(-0.5,4.2)
ax11.set_xlim(0,22)
ax11.set_ylim(0)

#set limits to the plot, as the lines of best fit will extend to areas that become irrelevant

handles, labels = ax11.get_legend_handles_labels()
ax11.legend(handles[::-1],labels[::-1],loc='upper right', numpoints=1)
#generates labels

plt.show()
#shows the completed plot
fig.savefig('magbreak8plot2exp.png')
#Saves the created plot

i=1
print("\n\n\n")
while i<n:
    if grad2[i]!=-10:
        print("For set", angular_velocity_headers[i-1],", the log of the data takes the form of two linear plots, defined as {:n} t + {:n} and {:n} t + {:n}." .format(grad[i],intercept[i],grad2[i],intercept2[i]))
        print("The data can then be plotted as two exponential plots, defined as e to the power of ({:n} t + {:n}) and e to the power of ({:n} t + {:n})." .format(a1[i],b1[i],a2[i],b2[i]))
        print("The second set of plots take over at roughly t = {:n}, but both sets of fits carry on past this as there is some overlap.\n".format(min(data.t[v[i]])))
    else:
        print("For set", angular_velocity_headers[i-1],", the log of the data takes the form of a linear plot, defined as {:n} t + {:n}.".format(grad[i],intercept[i]))
        print("The data can then be plotted as an exponential plot, defined as e to the power of ({:n} t + {:n}).\n".format(a1[i],b1[i]))
    i += 1
#prints the data from all the curve fits