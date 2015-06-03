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
# sfreq - sample frequency
# freqs - array of frequencies of interest
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
    tfr = tf.cwt_morlet(data, sfreq, freqs)
    return tfr

def tfwindow(data,sfreq, freqs, start=0,end=None, channel=0):
    window = np.expand_dims(data[channel,:],axis=0)
    tfr = morlet(window,sfreq,freqs)
    ntfr = baseline_normalize(tfr)
    if end != None:
        return ntfr[0,:,start:end]
    else:
        return ntfr[0,:,start:]

def find_SPW(data, kernel='kernel.npz', channel=0, plot=False):
    """
    Looks for sharp wave events in raw data.
    Uses matched filtering - Requires a .npz file to use as a kernel
    8 Hz - 30 Hz
    Restrict to 30 - 120 ms?
    -Ref: Buzsaki 1986, Brain Res.
    """
    #set_trace()
    with np.load(kernel) as match_data:
        ker = match_data['arr_0']
    corr = signal.correlate(data[channel,:], ker, mode='same')
    threshold = np.mean(corr) + 3*np.std(corr) # 2 sigma threshold
  #  SPW_location = np.where(np.abs(corr) > threshold)[0]
    if plot == True:
        plt.plot(np.arange(data.shape[1]),data[channel,:],\
                np.arange(data.shape[1]), np.abs(corr) > threshold)
        plt.show()
    return corr

def find_SPWR(data, filtered_data, fps, SPW, corr=None, channel=0):
    """
    fps - Frames per second. Required to construct window of size 120 ms
    """
    #set_trace()
    freqseries = pd.Series(filtered_data[channel,:])
    window = int(0.120 * fps)
    rolling_avg = np.nan_to_num(np.array(pd.rolling_mean(freqseries, window = window, center=True)))
    ripples = rolling_avg
    #ripples = np.abs(rolling_avg) > np.std(filtered_data[channel:])
  #  maxfreq = 1 / (0.030) # Hz
  #  N,Wn = signal.buttord(wp=maxfreq / (0.5*fps), ws=(1 / 0.020)/(0.5*fps), gpass=-10, gstop=20)
  #  b,a = signal.butter(N=N, Wn=Wn, btype='lowpass')
  #  ripples = signal.filtfilt(b,a,rolling_avg)
    freqs = np.arange(1,250,5)
    tfr = tfwindow(data,fps,freqs,channel=channel)
    SPWR = np.sum(tfr[29:,:].clip(min=0) * np.expand_dims(SPW,axis=0),axis=0)
    roll = np.nan_to_num(np.array(pd.rolling_mean(SPWR,window=window,center=True)))
    events = np.where(roll > 2.5*np.std(roll))[0]
    eventlist = np.split(events, np.where(np.diff(events) > 10)[0] + 1)
    return eventlist,tfr

def detect_SPWR(data, sfreq, channels='all'):
    """
    Detects SPWR in all channels
    """
    #set_trace()
    if channels == 'all':
        channels = xrange(data.shape[0])
    SPWRs = {}
    filtered_data = wave_filter(data,sfreq,passband='SWR')
    for ch in channels:
        SPW = find_SPW(data,channel=ch)
        SPWR,tfr = find_SPWR(data,filtered_data, sfreq, SPW, channel=ch)
        SPWRs[str(ch)]=SPWR
        print('Reading channel %d...' % ch)
    print('All channels read.')
    for ch in channels:
        candidates = map(lambda x:int(np.median(x)), SPWRs[str(ch)])
        print('Channel %d: %d likely sharp-wave ripple events' % (ch,len(candidates)))
    while(1):
        channel = input('Select a channel: ')
        event = 0
        while(1):
            print('For channel %d, likely sharp-wave ripple events at:' % channel)
            for i in range(len(candidates)):
                print('\t'+'%d. %d frames / %1.2f seconds' % (i,candidates[i],candidates[i]/float(sfreq)))
            event = input('Input an event to plot, or input -1 to return to channel selection: ')
            if event == -1: break
            width = int(sfreq / 2)
            start = candidates[event]-width/2
            end = candidates[event]+width/2
            plt.subplot(3,1,1)
            plt.plot(np.linspace(0,0.5,num=width),data[channel,start:end])
            plt.subplot(3,1,2)
            plt.plot(np.linspace(0,0.5,num=width),filtered_data[channel,start:end])
            plt.subplot(3,1,3)
            plt.imshow(tfr[:,start:end], aspect='auto', extent=[0,0.5,250,1])
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
