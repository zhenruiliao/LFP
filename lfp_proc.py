import numpy as np
import scipy.signal as signal
import mne
import pandas as pd
from mne import time_frequency as tf
from pudb import set_trace
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from intan_fixed import *

# data - 2d array of shape (channels, times)
# tfr - 3d array of shape (channels, frequencies, times)
#
#
#

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

def morlet(data, sfreq, freqs):
    tfr = tf.cwt.morlet(data, sfreq, freqs)
    return tfr

def find_SPW(data, kernel='kernel.npz', channel=0, plot=False):
    """
    Looks for sharp wave events in raw data.
    Requires a .npz file to use as a kernel
    8 Hz - 30 Hz
    Restrict to 30 - 120 ms?
    -Ref: Buzsaki 1986, Brain Res.
    """
    with np.load(kernel) as match_data:
        ker = match_data['arr_0']
    corr = signal.correlate(data[channel,:], ker, mode='same')
    threshold = np.mean(corr) + 3*np.std(corr) # 3 sigma threshold
  #  SPW_location = np.where(np.abs(corr) > threshold)[0]
    if plot == True:
        plt.plot(np.arange(data.shape[1]),data[channel,:],\
                np.arange(data.shape[1]), np.abs(corr) > threshold)
        plt.show()
    return corr > threshold

def find_SPWR(filtered_data, fps, SPW, corr=None, channel=0):
    """
    fps - Frames per second. Required to construct window of size 120 ms
    """
    set_trace()
    freqseries = pd.Series(filtered_data[channel,:])
    window = int(0.120 * fps)
    rolling_avg = np.array(pd.rolling_mean(freqseries, window = window, center=True))
    ripples = np.nan_to_num(rolling_avg)
  #  maxfreq = 1 / (0.030) # Hz
  #  N,Wn = signal.buttord(wp=maxfreq / (0.5*fps), ws=(1 / 0.020)/(0.5*fps), gpass=-10, gstop=20)
  #  b,a = signal.butter(N=N, Wn=Wn, btype='lowpass')
  #  ripples = signal.filtfilt(b,a,rolling_avg)
    SPWR = ripples * SPW
    events = np.where(SPWR != 0)[0]
    eventlist = np.split(events, np.where(np.diff(events) > 1)[0] + 1)
    return eventlist

def detect_SPWR(data, sfreq, channels='all'):
    """
    Detects SPWR in all channels
    """
    if channels == 'all':
        channels = xrange(data.shape[0])
    SPWRs = {}
    filtered_data = wave_filter(data,sfreq,passband='SWR')
    for ch in channels:
        SPW = find_SPW(data,channel=ch)
        SPWR = find_SPWR(filtered_data, sfreq, SPW, channel=ch)
        SPWRs[str(ch)]=SPWR
        print('Reading channel %d...' % ch)
    print('All channels read.')
    for ch in channels:
        candidates = map(lambda x:int(np.median(x)), SPWRs[str(ch)])
        print('For channel %d, likely sharp-wave ripple events at:' % ch)
        for i in range(len(candidates)):
            print('\t'+'%d. %d frames / %1.2f seconds' % (i,candidates[i],candidates[i]/float(sfreq)))
    while(1):
        channel,event = input('Input an event to plot in the form channel,event: ')
        width = int(sfreq / 2)
        plt.plot(np.linspace(0,0.5,num=width),data[channel,candidates[event]-width/2:candidates[event]+width/2])
        plt.show()

def wave_filter(data, fps, passband=None, channel=0, plot=False):
    # Filters from 150-250 Hz (SWR region)
    nyq = 0.5*fps
    if passband == 'SWR':
        wp = [150/nyq, 250/nyq]
        ws = [100/nyq, 300/nyq]
    elif passband == 'theta':
        wp = [4/nyq, 8/nyq]
        ws = [2/nyq, 10/nyq]
    elif passband == 'gamma':
        wp = [30/nyq, 50/nyq]
        ws = [20/nyq, 100/nyq]
    else:
        raise NameError('"%s" is not a recognized passband. ' +
                'The recognized passbands are SWR, theta, and gamma.' % passband)
    N = signal.buttord(wp=wp, ws=ws,gpass=-10,gstop=20)
    b,a = signal.butter(N=N[0], Wn=wp,btype='bandpass')
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
