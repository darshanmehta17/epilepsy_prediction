import numpy as np

import pyedflib
from utils import *

'''
    Inputs the signals from the file.
'''
def getSignalFromFile(file):
    # read file
    f = pyedflib.EdfReader(file)

    # create empty array for the signal
    signals = np.zeros((f.signals_in_file,f.getNSamples()[0]))
    
    # copy the signal from the file to the array
    for i in range(f.signals_in_file):
        signals[i,:] = f.readSignal(i)

    # close the file and free the variable
    f._close()
    del f
    return signals


'''
    Converts a given signal in time domain to frequency domain.
    Also chops off unwanted frequencies.
'''
def getFFT(signal, sampling_rate, freq_range):
    n = len(signal)
    k = np.arange(n)
    T = n / sampling_rate
    frq = k / T # two sides frequency range
    Y = np.fft.fft(signal) / n # fft computing and normalization
    
    # reshaping the fft signal
    Y = 10 * np.log10(np.square(np.absolute(Y)))
    
    freq_begin = int(round(freq_range[0] * n / sampling_rate))
    freq_end = int(round(freq_range[1] * n / sampling_rate)) + 1
    
    # chopping off unwanted frequencies
    Y = np.array(Y[freq_begin : freq_end])
    frq = np.array(frq[freq_begin: freq_end])
    
    return frq, Y


'''
    Creates 'm' equally spaced bandpass filters and returns the energy of the
    signal in them.
'''
def filter_energy(frq, signal, m):
    filter_size = (frq[-1] - frq[0]) / m # decides the size of each filter
    indices = []
    prev = 0
    cumsum = frq[0]

    # generating the border indices of the filters
    for i in range(m):
        cumsum = cumsum + filter_size
        new = index_of(frq, floor_half(cumsum))
        indices.append((prev, new))
        prev = new

    return np.array([np.max(signal[indice[0]: indice[1] + 1]) \
                    for indice in indices])


'''
    Stacks up FFTs of the sliding window to create an input for the SVM.
'''
def chi_T(signal, sampling_rate, starting_time, m, freq_range, window_size, \
            epoch_size):
    time_slots = range(starting_time, \
                    starting_time + window_size * epoch_size, epoch_size)
    chi_t = []
    for time in time_slots:
        sig_td = signal[:, time * sampling_rate : \
                        (time + epoch_size) * sampling_rate]
        x_t = []
        for channel in range(min(23, sig_td.shape[0])):
            frq, sig_fd = getFFT(sig_td[channel], sampling_rate, freq_range)
            x_t.append(filter_energy(frq, sig_fd, m))
        chi_t.append(x_t)
    return np.array(chi_t)


'''
    Takes an edf file as input and returns a data matrix suitable for input
    to the SVM by sweeping the sliding window across all time slots.
'''
def generateFileData(filename):
    
    # sampling rate
    sampling_rate = 256
    
    # number of filterbands
    m = 8
    
    # sliding window size
    window_size = 3
    
    # epoch size
    epoch_size = 2
    
    # required frequency range
    freq_range = [0.5, 25]
    
    # get signals from file
    signals = getSignalFromFile(filename)
    
    # to get the time size of signals
    signal_size = signals.shape[1] / sampling_rate
    
    return np.array([chi_T(signals, sampling_rate, time, m, freq_range, window_size, epoch_size) \
                     for time in range(int(np.floor(signal_size)) - window_size * epoch_size)])
        

def main():
    # name of the file to read
    filename = './chbmit/chb01/chb01_01.edf'

    # generate the data
    data = generateFileData(filename)

if __name__ == '__main__':
    import sys
    sys.exit(int(main() or 0))
