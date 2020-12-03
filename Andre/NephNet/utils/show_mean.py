import argparse
import torch
import math
import argparse
import os, os.path, shutil
import pathlib
import random
import pandas as pd
import numpy as np
import h5py
import random
print("Modules loaded")

filename = "/home/awoloshu/Desktop/NephNet/data/f33f44f59_combine_all/dataset_f33f44f59.h5"

with pd.HDFStore(filename) as f:
    print()
    print("Successfully opened " + filename)
    print("===============")
    keys = list(f.keys())
    print("{0} keys in this file: {1}".format(len(keys), keys))
    print(f['Metadata'])
    meta = f['Metadata']
    print("Training images: " + str(meta['TrainingNum']))
    print("Testing images: " + str(meta['TestingNum']))

    print("Training image mean = " + str(meta['TrainingMean'].values) + " and std = " + str(meta['TrainingStd'].values))

    test_data = f['test_data'].to_numpy()
    test_mean = np.mean(np.mean(test_data, axis = 0))
    test_std = np.mean(np.std(test_data, axis = 0))
    print("Testing image mean = " + str(test_mean) + " and std = " + str(test_std))
