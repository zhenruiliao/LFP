from sys import path
path.insert(0, '/home/jack/code/analysis_ser_dev/app/bin')

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
        import pdb; pdb.set_trace()
    axs[channel%4].plot(data['analog'][:,channel])


conn.saveFigure('lfp_'+str(int((channel-1)/4)), fig)
