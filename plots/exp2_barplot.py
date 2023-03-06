#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 19:14:21 2023

@author: ubuntujuanma
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

Algorithm = ['Mean','Median','Trimmed 80', 'Trimmed 60']
IDS = [ 'ID 3', 'ID 4', 'ID 12', 'ID 19']


data = np.array([[73.2,42.6,71.2,73.5],
    [29.8,29.8,31.1,29.9],
    [42.6,9.5,30.7,34.8],
    [64.0,30.0,62.7,58.0]])

fig, ax = plt.subplots(1,1, figsize=(5,3))
ax.set_title("Data Injection in Group Worker ID 2 at 0.5 ratio")
ax.set_ylabel("FNR %")
ax.set_xlabel("Outsiders workers")

colors = ['blue', 'orange', 'green', 'red']

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

pos=0
for i in range(0,4):
    for j in range(0,4):
        
        ax.bar(pos,data[i,j],color=colors[j])
        pos+=1
    pos+=1
    
ax.legend(Algorithm)
ax.set_xticks(ticks=[1.5, 6.5, 11.5,16.5])
ax.set_xticklabels(IDS)

fig.savefig('exp2_barplot_user2.pdf',bbox_inches="tight") 
