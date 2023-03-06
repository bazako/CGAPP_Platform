#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 19:15:06 2022

@author: ubuntujuanma
"""


import matplotlib.pyplot as plt
import numpy as np


dataModels=[
    
    [67.18366297, 72.30688761], 
    [60.77004886, 66.31701278],
    [74.48181824, 77.80243457],
    [63.18503556, 69.5774786],
    [77.66360086, 77.78172876],
    [64.49622818, 70.3058558],
    [64.65391378, 73.09063762],
    [56.84233433, 65.31156913],
    [79.52221315, 80.46476195],
    [60.61647149, 68.05927097],
    [76.92990693, 78.44643236],
    [60.25228948, 67.29296312],
    [79.97157761, 80.55494947],
    [62.09179813, 69.31524781],
    [78.38155699, 78.94209391],
    [61.52068699, 68.61357652]]



names=[' ','ABOD',
    ' ','IF',
    ' ','KNN',
    ' ','OCSVM',
    ' ', 'Autoenconder',
    ' ', 'Autoenconder 2 layers',
    ' ', 'A. Variational',
    ' ', 'A. Variational 2 layers']

percent = 0.2
colors=[
       [0.89411765, 0.10196078, 0.10980392, 1.        ], [0.89411765, 0.10196078, 0.10980392, percent        ],
       [0.21568627, 0.49411765, 0.72156863, 1.        ], [0.21568627, 0.49411765, 0.72156863, percent        ],
       [0.30196078, 0.68627451, 0.29019608, 1.        ], [0.30196078, 0.68627451, 0.29019608, percent        ],
       [0.59607843, 0.30588235, 0.63921569, 1.        ], [0.59607843, 0.30588235, 0.63921569, percent        ],
       [1.        , 1.        , 0.2       , 1.        ], [1.        , 1.        , 0.2       , percent        ],
       [0.65098039, 0.3372549 , 0.15686275, 1.        ], [0.65098039, 0.3372549 , 0.15686275, percent        ],
       [0.96862745, 0.50588235, 0.74901961, 1.        ], [0.96862745, 0.50588235, 0.74901961, percent        ],
       [0.6       , 0.6       , 0.6       , 1.        ], [0.6       , 0.6       , 0.6       , percent        ]]

category_colors = plt.get_cmap('Set1')(
        np.linspace(0, 1.0, 8))

numModels=len(dataModels)
pos = np.arange(len(dataModels))

fig, ax1 = plt.subplots(1, 1,figsize=(5, 3))
#fig.suptitle('Intervals confidence for Sensors and Statistics individual Models', fontsize=16)

ax1.set_title('Confident Intervals at 95%', fontsize=16)
ax1.axvline(x = 60 , color = 'grey', linestyle='--', linewidth = 0.5)
ax1.axvline(x = 65 , color = 'grey', linestyle='--', linewidth = 0.5)
ax1.axvline(x = 70 , color = 'grey', linestyle='--', linewidth = 0.5)
ax1.axvline(x = 75 , color = 'grey', linestyle='--', linewidth = 0.5)
ax1.axvline(x = 80 , color = 'grey', linestyle='--', linewidth = 0.5)

rects = ax1.barh(pos, [(dataModels[k][1] - dataModels[k][0]) for k in range(0,numModels)],
                     align='center',
                     left=[dataModels[k][0] for k in range(0,numModels)],
                     height=0.9,
                     # tick_label=names,
                     color=colors)


rects = ax1.barh(pos, [0.05 for k in range(0,numModels)],
                     align='center',
                     left=[(dataModels[k][0]+dataModels[k][1])/2 - 0.025 for k in range(0,numModels)],
                     height=0.8,
                     tick_label=names,color='k')



ax1.set_xlabel('Accuracy (percent)', fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=14)

fig.savefig('IntervalConfidenceExperiment1.pdf',bbox_inches="tight")