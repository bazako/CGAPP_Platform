#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 18:27:32 2023

@author: ubuntujuanma
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

x = [0,
    0.5,
    1,
    2,
    5,
    8,
    10,
    20,
    30,
    40,
    50,
    60,
    70,
    80,
    90,
    100]

Labels = ['0%',
    '0.5%',
    '1%',
    '2%',
    '5%',
    '8%',
    '10%',
    '20%',
    '30%',
    '40%',
    '50%',
    '60%',
    '70%',
    '80%',
    '90%',
    '100%']

# Recall
TPR = [97.59,
    53.34,
    53.01,
    52.58,
    51.51,
    49.81,
    50.80,
    48.60,
    38.73,
    39.96,
    37.34,
    27.41,
    32.35,
    24.80,
    23.49,
    15.85]

# Specifity 
TNR = [87.63,
    99.99,
    99.99,
    99.99,
    99.99,
    99.99,
    99.99,
    100.00,
    100.00,
    100.00,
    100.00,
    100.00,
    100.00,
    100.00,
    100.00,
    100.00]



fig, ax = plt.subplots(1,1, figsize=(5, 3))
ax.set_title("TPR and TNR for a Data Pertubation Attack with \n only a compromised worker using group distribution ")
ax.set_ylabel("%")
ax.set_xlabel("Percent of samples perturbed")

ax.plot(x,TPR, linestyle='solid', marker='o',linewidth=2, markersize=4)
ax.plot(x,TNR, linestyle='solid', marker='o',linewidth=2, markersize=4)

ax.legend(['TPR','TNR'])
ax.set_xticks(ticks=[0,                  10,    20,    30,    40,    50,    60,    70,    80,    90,    100])
ax.set_xticklabels(['0%',                  '10%',    '20%',    '30%',    '40%',    '50%',    '60%',    '70%',    '80%',    '90%',    '100%'])

fig.savefig('exp3_globalPoisoning_TPR-TNR.pdf',bbox_inches="tight") 