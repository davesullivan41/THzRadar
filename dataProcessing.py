from __future__ import print_function, division
import numpy as np
import scipy as sp
from scipy import signal
from scipy import interpolate 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
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
# 35  34  33  32  31  30  29  28  27
# 36  37  38  39  40  41  42  43  44
# .. 
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
# Arrange the waveforms appropriately -- possibly not necessary
########################################

########################################
# Find reference point for each transmitter
########################################


# distance between transmitter 0 and receiver
# directDistance = np.sqrt((0.118+xtx)**2+(0.014+ytx)**2+0.007**2)
# index of the max voltage - i.e. when the transmitted signal directly impacts the reciever
referenceVoltageIndexAll = voltage_raw[:,0]*0
for i in range(0,len(voltage_raw[:,0])):
	referenceVoltageIndexAll[i] = np.where(voltage_raw[i,:] == max(voltage_raw[i,:]))[0]

# Voltage indices follow the following pattern:
# 0	  1   2   3   4   5   6   7   8
# 17  16  15  14  13  12  11  10  9
# 18  19  20  21  22  23  24  25  26
# 35  34  33  32  31  30  29  28  27
# 36  37  38  39  40  41  42  43  44
# ... 
# 72  73  74  75  76  77  78  79  80 
# indices of transmitters we want to use
# 3x3
# indicesTX = np.matrix([[0,4,8],[36,40,44],[72,76,80]])
# 5x5
# indicesTX = np.matrix([[0,2,4,6,8],[18,20,22,24,26],[36,38,40,42,44],[54,56,58,60,62],[72,74,76,78,80]])
# 9x9
indicesTX = np.matrix([np.arange(0,8),np.arange(17,9,-1),np.arange(18,26),np.arange(35,27,-1),
			np.arange(36,44),np.arange(53,45,-1),np.arange(54,62),np.arange(71,63,-1),np.arange(72,80)])#[18,20,22,24,26],[36,38,40,42,44],[54,56,58,60,62],[72,74,76,78,80]])
print(indicesTX.shape)

# Get the voltage index at each of those indices
referenceVoltageIndex = referenceVoltageIndexAll[indicesTX]

# Coordinates of transmitters relative to transmitter 0 -- 3x3 grid
txInRow = len(indicesTX)
[xtx,ytx] = np.arange(0.,0.04,0.04/(txInRow-1)),np.arange(0.,-0.04,-0.04/(txInRow-1))
xtx = xtx 
ytx = ytx 



# Window we are trying to analyze 
# TODO: modify Z to analyze entire 3D window
X = np.arange(-0.10,0.15,0.002)
Y = np.arange(-0.15,0.15,0.002)
Z = 0.243

# Create matrices that hold the x and y displacement for each coordinate in image
Xi,Yi = np.meshgrid(X,Y)
# r2 = np.sqrt((Xi-0.118)**2 + (Yi+0.014)**2 + (Z-0.099)**2)


# For each antenna position, find distance to each point in grid
# Distance from each transmitter to each point
distancesTX = np.empty([len(X),len(Y),len(xtx),len(ytx)])
# Distance from each point to the transmitter
distancesRX = np.empty([len(Y),len(X)])
# Loop over every position x,y in the viewing area
for k in range(0,len(X)):
	for l in range(0,len(Y)):
		# Calculate the distance between each point in the grid and the receiver
		distancesRX[l,k] = np.sqrt((X[k]-0.118)**2 + (Y[l]+0.014)**2 + (Z-0.099)**2)
# 		# Loop over ach antenna position
		for i in range(0,len(xtx)):
			for j in range(0,len(ytx)):
				# Caculate the distance between each point in the grid and each transmitter
				distancesTX[k,l,i,j] = np.sqrt((xtx[i]-X[k])**2 + (ytx[j]-Y[l])**2 + (Z-0.106)**2)


## Caculate direct distance between transmitter and receiver
direct = np.empty([len(xtx),len(ytx)])
# Loop over each tx position
for i in range(0,len(xtx)):
	for j in range(0,len(ytx)):
		direct[i,j] = np.sqrt((0.118 - xtx[i])**2 + (0.014 + ytx[j])**2 + 0.007**2)
print('Distances calculated')

# discretizationsz
dis_step = 5e-3
dis_time = time[1]-time[0]
speed = 3e8

windowSize = 20

amplitude_sum = np.empty([len(Y),len(X)])
tempAmplitude = np.zeros(windowSize*2)
# For each position in the window, sum up waveforms from each tx position over a small window
for k in range(0,len(X)):
	for l in range(0,len(Y)):
		# amplitude_sum[l,k] = 0.
		tempAmplitude = tempAmplitude * 0.
		for i in range(0,len(xtx)):
			for j in range(0,len(ytx)):
				# Find the difference in distances for this tx pos
				diff_cont = (distancesTX[k,l,i,j] + distancesRX[i,j] - direct[i,j])/speed
				# diff_disc = round(diff_cont/dis_time)
				time_of_signal = int(round(referenceVoltageIndex[i,j] + diff_cont/dis_time))
				if(time_of_signal < windowSize):
					tempAmplitude += [np.zeros(windowSize-time_of_signal), voltage[indicesTX[i,j],range(0,time_of_signal+windowSize)]]#range(time_of_signal-100,time_of_signal+100)])
				elif (time_of_signal > timeLength-windowSize):
					tempAmplitude += [voltage[indicesTX[i,j],range(time_of_signal-windowSize,timeLength+1)], np.zeros(windowSize+time_of_signal-timeLength)]
				else:
					tempAmplitude += voltage[indicesTX[i,j],range(time_of_signal-windowSize,time_of_signal+windowSize)]
		amplitude_sum[l,k] = abs(sum(tempAmplitude))
	print('Row Completed')




########################################
# Test plot
########################################
fig = plt.figure()
plt.pcolormesh(Xi,Yi,amplitude_sum)
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(Xi,Yi,amplitude_sum,rstride=1, cstride=1, cmap=cm.coolwarm)
# plt.plot(X,Y,distancesRX)
# plt.title('Distance to receiver')
plt.show()