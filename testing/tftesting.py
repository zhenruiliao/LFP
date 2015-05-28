import numpy as np
import scipy as sp
import mne
from mne import time_frequency as tf
from pudb import set_trace
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from intan_fixed import *

def view(tfr, freqs, raw, channel=0, start=0, stop=None, width=250, fmax=None, symmetrize=True):
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
        pwrmax = max(pwr)
        pwrmin = min(pwr)
        vmin = min(pwrmin, -pwrmax)
        vmax = max(pwrmax, -pwrmin)
        norm = Normalize(vmin=vmin, vmax=vmax)

    plt.subplot(2,1,1)
    plt.imshow(pwr, extent=[start,stop,fmax,0], interpolation='nearest',
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
        aspect = 'auto')
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
    normalized_tfr = np.log(pwr / blf)
    return normalized_tfr

set_trace()
data_dict = np.load('recording_141202_000511_downsampled.npz')
data = data_dict['down_data']
data = np.transpose(data) # Transpose data to (leads, times)
sfreq = data.shape[1]/240.0 # Recording time = 240 s
freqs = np.asarray(range(2,50,1))
tfr = tf.cwt_morlet(data, sfreq, freqs)
# View tfr
view(tfr,freqs,data)
# Show channels 1,2,3,4,16
show_channels(data,channels = [1,2,3,4,16])
print('End of test.')

