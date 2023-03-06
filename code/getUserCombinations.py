#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 08:37:06 2022

@author: ubuntujuanma
"""

from itertools import combinations

def getUserCombinations( IDUsers, userCombination='Workers'):
    
    if userCombination=='Auto':
        groupUsers = [16, 6, 10, 5, 15, 2]
        impostorUsers = [4,12,19]
        middleUsers = [1,3,8,11,13,14,18,20,21]
        return groupUsers, impostorUsers, middleUsers
    
    elif userCombination=='Workers':
        groupUsers = [21,1,8,11,20,2]
        impostorUsers = [3,4,12,19] 
        middleUsers = [5,6,10,13,14,15,16,18]
        return groupUsers, impostorUsers, middleUsers

    else:
        raise Exception ("User Combination Not Valid")

    
        
        