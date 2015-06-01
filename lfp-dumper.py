import os.path
import csv
import itertools as it
from intan_fixed import read_data

#directory = ''
infile = 'gt1962_2_d1_150528_115815.int'
outfile = os.path.splitext(infile)[0] + '.csv'
#data = read_data(os.path.join(directory,infile))
data = read_data(infile)
data = data['analog'].T

with open(outfile, 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter='\t')
    for row in it.izip(*data):
        writer.writerow(row)
