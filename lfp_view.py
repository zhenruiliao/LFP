import numpy as np
import scipy.signal as signal
import mne
from mne import time_frequency as tf
from pudb import set_trace
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from intan_fixed import *

def view_tfr(tfr, freqs, raw, sfreq, channel=0, start=0, stop=None, width=250, fmin=None, fmax=None, symmetrize=True):
    """
    Parameters:
    TFR - power array. If convolution data is passed, power will be extracted automatically
        BASELINE NORMALIZATION HIGHLY RECOMMENDED!
    freqs - frequencies of wavelet
    sfreq - sample frequency
    raw - EEG raw data
    channel - Channel to view. Default 0
    start - Start time to view. Default 0
    width - Width of interval to view. Default 250 frames
    stop - Stop time to view. Default start + width
    symmetrize - Set to False if not using dB baseline normalization. Default True
    """
    if stop == None:
        stop = start+width
    if fmin == None:
        fmin = np.amin(freqs)
    if fmax == None:
        fmax = np.amax(freqs)
        if fmax > 0.5*sfreq: print 'Warning: max frequency exceeds Nyquist rate'
    pwr = np.squeeze(tfr[channel,np.where(freqs==fmin)[0][0]:np.where(freqs==fmax)[0][0],start:stop])
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
    plt.imshow(pwr, extent=[start,stop,fmax,fmin], interpolation='nearest',
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
        aspect = 20/9.0)
    plt.show()

def show_channels(data,channels = None):
    if channels == None:
        channels = xrange(data.shape[0])
    for i in xrange(len(channels)):
        plt.subplot(len(channels),1,i+1)
        plt.plot(data[channels[i],:])
    plt.show()
