import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

data = pd.read_excel('al0.xlsx', skiprows=1) #The file must have the time column headed with t, and the voltage columns headed with R
column_headers = list(data.columns) 
filter = re.compile(r'[R]\d+')
volt_headers = [header for header in column_headers if filter.findall(header)]
#These will read the excel sheet holding the data values- I have lumped together plots of the same magnetic field by taking the time column and the voltage columns and putting them all into one document

fig = plt.figure(figsize=(20, 10))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

def line(x, m, c):
    return m*x+c
def line2(x,a,b):
    return np.exp((x*a)+b)
#Defines the lines that the curve_fit will fit to- line2 function is defined according to log rules (I might have to tighten these up though)



n=1

for measurement in volt_headers:
    c=(np.sin(3*n)**2,np.cos(5*n)**2, abs(np.cos(5*n+3*np.pi/4))) #sets the colour for the plot- This set of values gives a nice rainbow for lots of values, but is just two different shades of blue if only using 2 sets
    v  = data[measurement].values
    w = []     #stores any found periods
    t_w = []    #stores the time associated with each period
    tempt = 0       # temporary values for immediate comparison
    i=0
    
    while i < len(data.t):   #for the entire set of V
        
        if abs(data.t[i]) >= 10:     #ignores any values for which |t| > 10 - hash out for full set
           i+=1
           continue
           #print("v = {}, i = {}".format(v[i],i))
        if v[i] < 38:                                           #checks if the voltage is below a certain threshold, may need altering depending on dataset

            #print("\n    v < 10!    t = {}\n".format(t[i]))
            if abs(v[i+1]-v[i]) > 1:                            #checks if the difference to the voltage ahead of it is greater than 1 voltage - implies the end of the step
            
                #print("     volt step > 1\n")
                if tempt == 0:                                  #if the temp holder is empty, treat the found value as the first reference point 
                    tempt = data.t[i]
                else:
                    if data.t[i]-tempt > 0.02:                       #otherwise if the difference in time to the temp holder is greater than a certain interval, store the time diff. - threshold may need altering depending on dataset

                        #print("     time step > 0.01\n")
                        w.append(2*np.pi/(data.t[i]-tempt))
                        t_w.append(data.t[i])
                        tempt=data.t[i]                              #note this time as the new reference point
    
        i+=1
    
    popt1, pcov1 =curve_fit(line, t_w, np.log(w))
    popt2, pcov2 =curve_fit(line2, t_w, w)
    times=np.linspace(min(t_w),max(t_w),1000)
    curve_val1=line(times,popt1[0],popt1[1])
    curve_val2=line2(times,popt2[0],popt2[1]) #These calculate a curvefit for the log plot and regular plot, and give a set of values that can be plotted as the fit
    
    
    ax1.plot(t_w,np.log(w),
             color=c,
             marker='x',
             linestyle='none')      
    ax2.plot(t_w,w,
             color=c,
             marker='x',
             linestyle='none')
    ax1.plot(times,curve_val1,
             color=c,
             linestyle='--')
    ax2.plot(times,curve_val2,
             color=c,
             linestyle='--')
    print("For set", volt_headers[n-1],", the log of the data takes the form of a linear plot, defined as {:n} t + {:n}.".format(popt1[0],popt1[1]))
    print("The data can then be plotted as an exponential plot, defined as e to the power of ({:n} t + {:n}).\n".format(popt2[0],popt2[1]))
    n+=1 #moves to next measurement- this isn't necessary for the data, but it is to calculate the new colour


plt.ylabel("Ln(τ)")
plt.xlabel("time / s")
plt.show()

fig.savefig('magbreak0.png') #saves the resulting graph
