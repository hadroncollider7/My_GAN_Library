# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 16:35:55 2022

@author: longc
"""
import numpy as np
#%%
a = np.array(range(10))
print(a)
print(type(a))
print(a.shape)
print(a.dtype)
#%%
for step in range(50):
    print('a[{0}] = {1}'.format(step, a[step % 10]))
#%%
batch_size = 4
start = 0
for step in range(50):
    print('start = {1}, len(a) - start = {0}'.format(a.shape[0] - start, start))
    if (start > a.shape[0] - batch_size):
        real = a[start:start + (a.shape[0] - start)]
        real = np.concatenate([real, a[:a.shape[0] - start]])
        print(real)
        start = a.shape[0] - start
    else: 
        real = a[start:start + batch_size]
        print(real)
        start += batch_size
    if a.shape[0] - start == 0:
        start = 0
        # {shuffle array here}
