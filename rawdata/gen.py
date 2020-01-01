import os
import sys
import logging
import numpy as np
from multiprocessing import Process

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import *

def work(pid, pcnt):
    ds = DataGenerator(
        'csv/unbalanced_train_segments_filtered.csv',
        'Video/', 'Audio/', pid, pcnt, 0x19260817
      )
    data = []
    sub = 0

    for img, aud, out in ds:
        data.append((img, aud, out))

        if len(data) == 900:
           np.save("Data/data_part{}sub{}.npy".format(ds.pid, sub), data)
           logging.info('[pid %s] saved sub%s', ds.pid, sub)
           sub += 1

           del data
           del ds.saved_audios
           ds.saved_audios = dict()
           data = []

    if len(data) != 0:
        np.save("Data/data_part{}sub{}.npy".format(ds.pid, sub), data)

logging.basicConfig(level=logging.INFO)

n_proc = 60
proc = []
for pid in range(n_proc):
    proc.append(Process(target=work, args=(pid, n_proc)))

for p in proc:
    p.start()
for p in proc:
    p.join()
