import os.path
import csv
import itertools as it
from intan_fixed import read_data

directory = '/data/Gergo/LFP_DATA/gt_lfp1/gt_lfp01_2015-04-20-14h00m15s.int/'
infile = 'recording_150420_140010.int'
outfile = os.path.splitext(infile)[0] + '.csv'
data = read_data(os.path.join(directory,infile))

data = data['analog'].T

with open(outfile, 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter='\t')
    for row in it.izip(*data):
        writer.writerow(row)
