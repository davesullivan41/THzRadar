from __future__ import print_function, division
import numpy as np
import scipy as sp
from scipy import signal
from scipy import interpolate 
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

# Voltage indices follow the following pattern:
# 0	  1   2   3   4   5   6   7   8
# 17  16  15  14  13  12  11  10  9
# 18  19  20  21  22  23  24  25  26
# ...
# 72  73  74  75  76  77  78  79  80 
### This needs to be fixed.

########################################
# Normalize and filter the signal
########################################

timeLength = len(time)

# modify the signal so that the average voltage from any tx position is zero
for k in range(0,81):
	voltage[k,:] = voltage[k,:] - voltage[k,:]/timeLength
	voltage_raw[k,:] = voltage_raw[k,:] - voltage_raw[k,:]/timeLength

# generate a gaussian window of length 10, with sigma = 32
gaussFilter = sp.signal.gaussian(10,32)

# double filter both voltage and raw voltage
for k in range(0,81):
	# convolve the voltage signal with the gaussian window, twice
	voltage[k,:] = sp.signal.convolve(voltage[k,:],gaussFilter)
	voltage[k,:] = sp.signal.convolve(voltage[k,:],gaussFilter)
	voltage_raw[k,:] = sp.signal.convolve(voltage[k,:],gaussFilter)
	voltage_raw[k,:] = sp.signal.convolve(voltage[k,:],gaussFilter)

# normalize so the max voltage from each tx location is the same
globalMax = max(voltage)
for k in range(0,81):
	voltage[k,:] = voltage[k,:] * globalMax/max(voltage[k,:])

########################################
# Interpolate -- why? -- lets see what happens without it
########################################

# for k in range(0,81):
# 	interpFun = sp.interpolate.interp1d(range(0,4096),voltage[k,:])
# 	for j in range(0,4096*8):


########################################
# Arrange the waveforms appropriately, what is with all the zeros?
########################################
###

direct = np.sqrt(0.118**2+0.014**2+0.007**2)
reference = voltage_raw[0,:].index(max(voltage_raw[0,:]))

########################################
# Test plot
########################################

plt.figure(2)
plt.plot(voltage_filter1)
plt.show()