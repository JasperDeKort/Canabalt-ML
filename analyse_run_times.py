# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 08:13:40 2017

@author: jaspe
"""

import numpy as np
import pandas as pd
import os


def load_run_times():
    df = pd.DataFrame()
    for i in range(4, 20):
        if os.path.isfile('run_times_{}.npy'.format(i)):
            print('run times {} exists. loading data'.format(i))
            df['run{}'.format(i)] = pd.Series(np.load('run_times_{}.npy'.format(i)))
    print(df.head())
    return df

run_times_df = load_run_times()