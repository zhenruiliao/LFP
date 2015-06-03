import numpy as np
import scipy.signal as signal
import mne
from mne import time_frequency as tf
from lfp_view import *
from lfp_proc import *
from pudb import set_trace
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from intan_fixed import *

#data_dict = np.load('recording_141202_000511_downsampled.npz')
#data = data_dict['down_data']
#data = np.transpose(data) # Transpose data to (leads, times)
#sfreq = data.shape[1]/240.0 # Recording time = 240 s
#freqs = np.arange(2,250,5)
#tfr = tf.cwt_morlet(data, sfreq, freqs)
mydata,fps = readint('gt1962_2_d1_150528_115815.int')
data,fps = downsample(mydata,fps=fps)

detect_SPWR(data,fps,channels=[3])

#filtered_data = wave_filter(data,fps, passband='SWR')
#corr = find_SPW(data,channel=3)
#find_SPWR(data,filtered_data,fps,corr,channel=3)
#freqs = np.arange(2,500,2)
#pwr = baseline_normalize(tfr)
#tfr = tf.cwt_morlet(data,fps,freqs)

# Baseline normalize
#tfr = baseline_normalize(tfr)

# View tfr
#view(tfr,freqs,data)
# Show channels 1,2,3,4,16
#show_channels(data,channels = [1,2,3,4,16])
print('End of test.')
