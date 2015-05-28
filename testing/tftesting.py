import numpy as np
import scipy.signal as signal
import mne
from mne import time_frequency as tf
from pudb import set_trace
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from intan_fixed import *

def view_tfr(tfr, freqs, raw, channel=0, start=0, stop=None, width=250, fmax=None, symmetrize=True):
    """
    Parameters:
    TFR - power array. If convolution data is passed, power will be extracted automatically
        BASELINE NORMALIZATION HIGHLY RECOMMENDED!
    freqs - frequencies of wavelet
    raw - EEG raw data
    channel - Channel to view. Default 0
    start - Start time to view. Default 0
    width - Width of interval to view. Default 250 frames
    stop - Stop time to view. Default start + width
    symmetrize - Set to False if not using dB baseline normalization. Default True
    """
    if stop == None:
        stop = start+width
    if fmax == None:
        fmax = tfr.shape[1]-1
    pwr = np.squeeze(tfr[channel,:fmax,start:stop])
    if np.iscomplexobj(pwr):
        print('Warning: Complex array passed. Extracting powers automatically...')
        pwr = np.real(pwr*np.conj(pwr))
    # Symmetrize colorbars
    norm = None
    if symmetrize:
        pwrmax = np.amax(pwr)
        pwrmin = np.amin(pwr)
        vmin = min(pwrmin, -pwrmax)
        vmax = max(pwrmax, -pwrmin)
        norm = Normalize(vmin=vmin, vmax=vmax)

    plt.subplot(2,1,1)
    plt.imshow(pwr, extent=[start,stop,np.amax(freqs),0], interpolation='nearest',
            aspect='auto', norm=norm)
    plt.colorbar()
    plt.subplot(2,1,2)
    plt.plot(np.array(range(start,stop)),np.squeeze(raw[channel,start:stop]))
    plt.show()

def dBplot(tfr, freqs, channel=0):
    chan = tfr[channel,:,:]
    rowsums = np.sum(chan*np.conj(chan),axis=1)
    dBs = 10*np.log(rowsums)
    plt.plot(freqs, dBs)
    plt.show()

def tfrview(tfr, channel=0, averaging_window = 500,fmax=None):
    if fmax==None:
        fmax = tfr.shape[1]
    data = np.squeeze(tfr[channel,:fmax,:])
    data = np.real(data*np.conj(data))
    rows = np.vsplit(data,data.shape[0])
    newrows = []
    for row in rows:
        blocked_row = np.reshape(row,(-1,averaging_window))
        block_avgs = np.sum(blocked_row,axis=1, dtype=float) / averaging_window
        newrows.append(np.squeeze(block_avgs))
    newdata = np.vstack(newrows)
    plt.imshow(newdata, extent=[0,data.shape[1], fmax,0], interpolation='nearest',
        aspect = 2020/9.0)
    plt.show()

def show_channels(data,channels = None):
    if channels == None:
        channels = xrange(data.shape[0])
    for i in xrange(len(channels)):
        plt.subplot(len(channels),1,i+1)
        plt.plot(data[channels[i],:])
    plt.show()

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

def HPF(data, N, Wn=70.0/250, channel=0, plot=True):
    b,a = signal.butter(N=N, Wn=Wn,btype='high')
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


data_dict = np.load('recording_141202_000511_downsampled.npz')
data = data_dict['down_data']
data = np.transpose(data) # Transpose data to (leads, times)
sfreq = data.shape[1]/240.0 # Recording time = 240 s
freqs = np.arange(2,250,5)
tfr = tf.cwt_morlet(data, sfreq, freqs)
set_trace()
# Baseline normalize
tfr = baseline_normalize(tfr)

# View tfr
view(tfr,freqs,data)
# Show channels 1,2,3,4,16
show_channels(data,channels = [1,2,3,4,16])
print('End of test.')

