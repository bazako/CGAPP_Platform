#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 22 12:42:17 2022

@author: ubuntujuanma
"""
import numpy as np
import pandas as pd

def getData_fromfileSesion(fileSesion):
    sensor=[]
    statistics=[]
    voiceNotes=[]
    CallRecordings=[]    
    fid=open(fileSesion,'r', encoding = "ISO-8859-1") 
    
    lines=fid.readlines()
    for line in lines:
        #line = fid.readline()
        line=line.replace('\n','')
        #print(line)
        if not line:
            break
        
        lineSplit=line.split(',')
        numfields=len(lineSplit)
        second_field=lineSplit[1]
        
        
        if numfields>512:
            if second_field=='vn':
                # Voicenote 
                # voiceNotes.append(map(str.strip, line.split(',')))
                voiceNotes.append(line.split(','))
            else:
                # Call recorded
                CallRecordings.append(line.split(','))
            
        elif numfields>20:
            # App' statistic
            #data=map(str.strip, line.split(','))
            sensor.append(line.split(','))

        else:
            #data=map(str.strip, line.split(','))
            statistics.append(line.split(','))
        
        
        
        
    fid.close()
    
    sensor = np.array(sensor)
    statistics = np.array(statistics)
    voiceNotes = np.array(voiceNotes)
    CallRecordings = np.array(CallRecordings)
    
    dias=['lun.','mar.','mié.','jue.','vie.','sáb.','dom.']
    days=['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
    for idia in range(0,7):
        dia=dias[idia]
        day=days[idia]
        if sensor.shape[0]>0:
            sensor[sensor==dia]=str(idia)
            sensor[sensor==day]=str(idia)
        if statistics.shape[0]>0:
            statistics[statistics==dia]=str(idia)
            statistics[statistics==day]=str(idia)
    
    return sensor, statistics, voiceNotes, CallRecordings


def getData_fromFileUser(userfile, datatype=None, useTempInfo=False, return_timestamp=False):
    dataFrameUserTrain=pd.read_csv(userfile,header=None)
                    
    if not datatype=='voice':
        data = dataFrameUserTrain.values[:,1:]

        
        ####### !!!!!!!! OJO RECORTAR !!!!!!!!!! 
        if not useTempInfo:
            data = data[:,:-2]
    else:
        data = dataFrameUserTrain.values[:,2:]
    
    data = np.array(data,np.float64)
    
    if return_timestamp: 
        return data, dataFrameUserTrain.values[:,0]
    else:
        return data