#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 20:09:29 2022

@author: ubuntujuanma
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 19:15:06 2022

@author: ubuntujuanma
"""


import matplotlib.pyplot as plt
import numpy as np


dataModels=[
    
    
    [79.52221315,80.46476195],
    [68.76592279,70.24141933],
    [78.0780145,78.38037274],
    [76.92990693,78.44643236],
    [67.78502419,70.36615385],
    [76.02526654,77.21882542],
    [77.66360086,77.78172876],
    [70.57354161,70.57354161],
    [76.02442635,76.02476495],
    [79.97157761,80.55494947],
    [66.32153381,67.72617657],
    [75.49332327,75.86635129],
    [78.38155699,78.94209391],
    [68.44011475,69.7049969],
    [74.62835235,75.33935304]]





names=[
    ' ',
    ' ',
    'Autoencoder',
    '    ',
    '     ',
    'Autoencoder 2 layers',
    ' ',
    ' ',
    'KNN',
    ' ' ,
    ' ',
    'A. Variational',
    ' ',
    ' ',
    'A. Variational 2 layers']



percent = 0.2
colors=[
       [0.89411765, 0.10196078, 0.10980392, 1.        ],
       [0.21568627, 0.49411765, 0.72156863, 1.        ],
       [0.30196078, 0.68627451, 0.29019608, 1.        ],
       [0.89411765, 0.10196078, 0.10980392, 1.        ],
       [0.21568627, 0.49411765, 0.72156863, 1.        ],
       [0.30196078, 0.68627451, 0.29019608, 1.        ],
       [0.89411765, 0.10196078, 0.10980392, 1.        ],
       [0.21568627, 0.49411765, 0.72156863, 1.        ],
       [0.30196078, 0.68627451, 0.29019608, 1.        ],
       [0.89411765, 0.10196078, 0.10980392, 1.        ],
       [0.21568627, 0.49411765, 0.72156863, 1.        ],
       [0.30196078, 0.68627451, 0.29019608, 1.        ],
       ]

category_colors = plt.get_cmap('Set1')(
        np.linspace(0, 1.0, 8))

numModels=len(dataModels)
pos = np.arange(len(dataModels))

fig, ax1 = plt.subplots(1, 1,figsize=(5, 3))
#fig.suptitle('Intervals confidence for Sensors and Statistics individual Models', fontsize=16)

ax1.set_title('Confident Intervals at 95%', fontsize=16)

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
                     left=[(dataModels[k][0]+dataModels[k][1])/2 - 0.025  for k in range(0,numModels)],
                     height=0.8,
                     tick_label=names,color='k')



ax1.set_xlabel('Accuracy (percent)', fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=14)

fig.savefig('IntervalConfidenceExperiment2.pdf',bbox_inches="tight")