import numpy as np
import scipy.signal as signal
import mne
from mne import time_frequency as tf
from pudb import set_trace
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from intan_fixed import *

def baseline_normalize(tfr, start=0, stop=None,channel='all', window=50, alg='dB'):
    """
    Baseline window: start (default frame 0)
        Window: Size of window
    Algorithms available: dB (decibel change from baseline), prct (percent change from baseline)
        Z (Z transform)
    """
    if stop==None:
        stop = start + window
    # tfr must be in form (channels, times)
    # Returns normalized power
    pwr = np.real(tfr*np.conj(tfr))
    baseline_f = np.sum(pwr[:,:,start:stop],axis=-1) / (stop - start)
    blf = np.expand_dims(baseline_f,axis=-1)
    if alg=='dB':
        normalized_tfr = 10*np.log(pwr / blf)
    elif alg=='prct':
        normalized_tfr = 100*(tfr - blf)/blf
    elif alg=='Z':
        normalized_tfr = (tfr - blf) / np.std(blf)
    else:
        raise NameError('"%s" is not a recognized algorithm. The available algorithms are dB, prct, and Z' % alg)
    return normalized_tfr

def SWR_filter(data, fps, passband=None, channel=0, plot=False):
    # Filters from 150-250 Hz (SWR region)
    nyq = 0.5*fps
    N = signal.buttord(wp=[150/nyq,250/nyq], ws=[100/nyq,300/nyq],gpass=-10,gstop=20)
    b,a = signal.butter(N=N[0], Wn=[150/nyq,251/nyq],btype='bandpass')
    filtered_data = signal.filtfilt(b,a,data)
    if plot==True:
        plt.subplot(2,1,1)
        plt.title('Original timeseries')
        plt.plot(data[channel,:])
        plt.subplot(2,1,2)
        plt.title('Filtered timeseries')
        plt.plot(filtered_data[channel,:])
        plt.show()
    return filtered_data

def readint(intfile):
    mydata = read_data(intfile)
    duration = mydata['duration']
    data = np.transpose(mydata['analog'])
    fps = data.shape[1] / float(duration)
    return data,fps

def downsample(data, fps=None, step=10):
    if fps==None:
        return data[:,::step]
    else:
        return data[:,::step],fps/float(step)
