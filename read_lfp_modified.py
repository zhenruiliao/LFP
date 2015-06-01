from sys import path
path.insert(0, '/home/jack/code/analysis_ser_dev/app/bin')


from scipy import signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os.path

import reportManager
from intan_fixed import read_data

directory = '/data/Gergo/LFP_DATA/gt_lfp1/gt_lfp01_2015-04-20-14h00m15s.int/'
infile = 'recording_150420_140010.int'
data = read_data(os.path.join(directory,infile))

conn = reportManager.dbConnection()
for channel in xrange(data['analog'].shape[1]):
    print 'plotting channel ' + str(channel)
    if channel%4 == 0:
        if channel != 0:
            conn.saveFigure('lfp_'+str(int((channel-1)/4)), fig)
        fig,axs = plt.subplots(nrows=4, figsize=(10,8))
    channel_data = data['analog'][:,channel]

    import pudb; pudb.set_trace()
    N=101
    nyq= 0.5 * 25000
    lowcut= 0.1/nyq
    highcut=5000/nyq
    Wn=[lowcut, highcut]
    btype='bandpass'
    b,a=signal.butter(N,Wn,btype=btype)
    channel_data=signal.filtfilt(b,3,channel_data)


    axs[channel%4].plot(channel_data)
    break


conn.saveFigure('mac_lfp_'+str(int((channel-1)/4)), fig)
