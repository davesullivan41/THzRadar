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
	voltage[k,:] = sp.signal.convolve(voltage[k,:],gaussFilter,mode='same')
	voltage[k,:] = sp.signal.convolve(voltage[k,:],gaussFilter,mode='same')
	voltage_raw[k,:] = sp.signal.convolve(voltage_raw[k,:],gaussFilter,mode='same')
	voltage_raw[k,:] = sp.signal.convolve(voltage_raw[k,:],gaussFilter,mode='same')

# normalize so the max voltage from each tx location is the same
globalMax = voltage.max()
for k in range(0,81):
	voltage[k,:] = voltage[k,:] * globalMax/max(voltage[k,:])

########################################
# Interpolate for higher resolution --- later
########################################
# plt.figure(1)
# plt.plot(voltage[0,:])
# plt.title('Original')

# voltageExpansion = np.zeros((81,len(voltage[0,:])*8))
# for k in range(0,81):
# 	interpFun = sp.interpolate.interp1d(range(0,4096),voltage[k,:])
# 	for j in range(0,4095*8):
# 		voltageExpansion[k,j] = interpFun(j/8.)
# 	print('Finished round '+str(k))
# 	# voltage[k,:] = np.copy(voltageExpansion)


########################################
# Arrange the waveforms appropriately
########################################

direct = np.sqrt(0.118**2+0.014**2+0.007**2)
reference = voltage_raw[0,:].index(max(voltage_raw[0,:]))

# discretizations
dis_step = 5e-3
dis_time = time[1]-time[0]
speed = 3e8

X = np.arange(-0.05:0.002:0.15)
Y = np.arange(-0.15:0.002:0.15)
Z = 0.243

image = zeros((len(X),len(Y)))

#### test using only 9 coordinates
dis_diff = zeros((3,3))
delay_diff = zeros((3,3))

Xi,Yi = meshgrid(X,Y)

r1 = np.sqrt(Xi**2 + Yi**2 + (Z-0.106)**2)
r2 = sqrt((Xi-0.118)**2 + (Yi+0.014)**2 + (Z-0.099)**2)

index = reference + round((r1+r2-direct)/speed/dis_time)-start+1

########################################
# Test plot
########################################

plt.figure(2)
plt.plot(voltageExpansion[0,:])
plt.title('Interpolated')
plt.show()