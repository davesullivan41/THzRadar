from __future__ import print_function, division
import numpy as np
import scipy as sp
from scipy import ndimage
from scipy import signal
import matplotlib.pyplot as plt
import os.path


########################################
# Load data
########################################

# load time
filename = 'time.txt'
if(os.path.isfile(filename)):
	time = np.loadtxt(filename)

# load voltage
filename = 'voltage.txt'
if(os.path.isfile(filename)):
	voltage = np.loadtxt(filename)
	# voltage[transmitter number, time]

# load raw voltage
filename = 'voltage_raw.txt'
if(os.path.isfile(filename)):
	voltage_raw = np.loadtxt(filename)
	# voltage_raw[transmitter number, time]


########################################
# Normalize and filter the signal
########################################

timeLength = len(time)

# modify the signal so that the average voltage from any tx position is zero
for k in range(0,81):
	voltage[k,:] = voltage[k,:] - voltage[k,:]/timeLength
	voltage_raw[k,:] = voltage_raw[k,:] - voltage_raw[k,:]/timeLength

# generate a gaussian window
gaussFilter = sp.signal.gaussian(10,32)

# double filter both voltage and raw voltage
for k in range(0,81):
	voltage[k,:] = sp.signal.convolve(voltage[k,:],gaussFilter)
	voltage[k,:] = sp.signal.convolve(voltage[k,:],gaussFilter)
	voltage_raw[k,:] = sp.signal.convolve(voltage[k,:],gaussFilter)
	voltage_raw[k,:] = sp.signal.convolve(voltage[k,:],gaussFilter)

########################################
# Interpolate -- why?
########################################


########################################
# Test plot
########################################

plt.figure(2)
plt.plot(voltage_filter1)
plt.show()