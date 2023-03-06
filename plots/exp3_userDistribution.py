#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 18:46:27 2023

@author: ubuntujuanma
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

x = [
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

TPR = np.array([[97.60,97.79,97.89,97.59,97.67,97.54],
    [97.58,97.88,97.74,97.11,97.71,97.52],
    [97.51,97.27,97.67,97.78,97.57,97.55],
    [97.39,97.07,97.43,97.28,97.58,97.69],
    [97.31,95.82,96.69,96.84,97.22,97.43],
    [97.20,96.02,97.08,96.28,96.64,97.90],
    [96.75,94.56,96.33,96.31,95.75,97.18],
    [97.29,95.01,95.62,95.53,95.15,96.62],
    [96.30,93.56,96.10,94.33,95.39,95.81],
    [95.78,94.86,94.81,95.32,94.10,96.55],
    [95.64,93.96,94.11,95.45,95.58,96.35],
    [95.37,94.48,95.44,94.69,95.10,96.13],
    [94.97,96.26,93.96,94.89,95.17,96.03],
    [94.64,97.22,94.29,94.41,95.28,95.43],
    [96.08,98.04,94.74,95.79,96.00,96.16]])


TNR = np.array([[89.06,87.10,88.56,88.47,88.45,89.16],
    [88.87,88.83,88.36,86.93,89.18,88.44],
    [87.77,88.03,89.29,88.85,88.98,89.50],
    [89.13,88.85,89.55,89.27,89.47,90.00],
    [89.21,89.86,89.18,89.97,89.85,90.50],
    [89.34,88.84,90.33,89.17,90.38,90.72],
    [89.56,90.24,90.65,90.18,91.88,92.56],
    [90.21,90.77,91.52,90.82,91.60,90.60],
    [89.99,90.48,91.70,92.42,88.39,92.90],
    [90.05,92.52,91.07,88.66,89.04,89.71],
    [90.00,90.04,88.84,86.29,84.83,86.02],
    [90.03,89.34,88.64,87.08,79.97,85.21],
    [90.59,89.17,87.26,85.93,77.84,71.41],
    [90.17,90.16,89.12,84.62,78.11,60.36],
    [90.28,86.44,85.96,74.76,63.67,46.16]])


fig, ax = plt.subplots(1,1, figsize=(5,3))
ax.set_title("Data Pertubation Attack using compromised worker distribution ")
ax.set_ylabel("TPR %")
ax.set_xlabel("Percent of samples perturbed")

for i in range(0,6):
    ax.plot(x,TPR[:,i], linestyle='solid', marker='o',linewidth=2, markersize=4)


ax.legend(['1 compromised worker','2 compromised workers','3 compromised workers','4 compromised workers','5 compromised workers','6 compromised workers'])
ax.set_xticks(ticks=[0,                  10,    20,    30,    40,    50,    60,    70,    80,    90,    100])
ax.set_xticklabels(['0%',                  '10%',    '20%',    '30%',    '40%',    '50%',    '60%',    '70%',    '80%',    '90%',    '100%'])
plt.ylim([90,100])

fig.savefig('exp3_workerPoisoning_TPR.pdf',bbox_inches="tight") 



fig, ax = plt.subplots(1,1, figsize=(5, 3))
ax.set_title("Data Pertubation Attack using compromised worker distribution ")
ax.set_ylabel("TNR %")
ax.set_xlabel("Percent of samples perturbed")

for i in range(0,6):
    ax.plot(x,TNR[:,i], linestyle='solid', marker='o',linewidth=2, markersize=4)


ax.legend(['1 compromised worker','2 compromised workers','3 compromised workers','4 compromised workers','5 compromised workers','6 compromised workers'])
ax.set_xticks(ticks=[0,                  10,    20,    30,    40,    50,    60,    70,    80,    90,    100])
ax.set_xticklabels(['0%',                  '10%',    '20%',    '30%',    '40%',    '50%',    '60%',    '70%',    '80%',    '90%',    '100%'])
# plt.ylim([90,100])

fig.savefig('exp3_workerPoisoning_TNR.pdf',bbox_inches="tight") 