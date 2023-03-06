#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 17:39:59 2022

@author: ubuntujuanma
"""
# -*- coding: utf-8 -*-

import glob
import os 
import numpy as np

from getData import getData_fromFileUser
from getUserCombinations import getUserCombinations
# import json
from progressbar import progressbar
import argparse
# from utils import calculate_eer, calculate_metrics
import models 
import random

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import tensorflow as tf
import tensorflow_federated as tff
import math
# import nest_asyncio
# nest_asyncio.apply()



random.seed(21712)
np.random.seed(21712)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('datatype', type=str, default='sensors', help='sensors, statistics or voice')
parser.add_argument('model', type=str)
parser.add_argument('--percent', type=float, default=0.0, help='Value of percent to include of a Backdoor Attack')
parser.add_argument('--numClientDenial', type=int, default=0, help='Num of bad clients, from 0 to 11')
parser.add_argument('--algorithmAgregation', type=str, default='mean', help='Algorithm select to do the aggregation')
parser.add_argument('--useTempInfo', type=bool, default=False, help='Verbose')
parser.add_argument('--verbose', type=bool, default=True, help='Verbose')



args = parser.parse_args()
print(args)
datatype = args.datatype 
modelName = args.model
verbose = args.verbose
useTempInfo = args.useTempInfo
algorithmAgregation = args.algorithmAgregation
percent = args.percent
numClientDenial = args.numClientDenial
typeCom = 'Workers'

#datatype = 'statistics'
#modelName = 'Autoencoder_1_16'
#verbose = True
#typeCom='clusterForced'

#pathToFiles='../data/14days/'
#useTempInfo=False

pathToFiles='data/14days/'


nameModel=modelName #str(model).split('.')[-1].replace("'","").replace('>','')


pathToScores='./scores/exp3_DenialService/' 
pathToScores+= 'percent'+str(percent)+'_'
pathToScores+= 'numClientDenial'+str(numClientDenial)+'_'
pathToScores+= nameModel+'/'
pathToScores+= 'algorithm'+algorithmAgregation


users=glob.glob(pathToFiles+'train/*'+datatype+'.txt')
users.sort()



# Users in groups of 10 persons
numIDuser = []
for kuser in range(0,len(users)):
    filetest = users[kuser]
    numIDuser.append(int(filetest.split('user')[1].split('_')[0]))

groupUsers, impostorUsers, middleUsers = getUserCombinations( numIDuser, userCombination=typeCom) 

os.makedirs(pathToScores,exist_ok=True)   
file_resume = pathToScores + '/resume_'+datatype+'.txt'
a_file=open(file_resume, "a")





    
auc=[]
precision=[]
recall=[]
F1_score=[]
tnr=[]
specificity=[]


min_data = [0,0,1001,0,1001,1001,1001,6,3,1001,4,0,0]
max_data = [11,6,1281,7,1280,1281,1281,1365,34,1246,399,4.52157e+10,2.78382e+09]



def client_data(userfile,datatype, useTempInfo, scaler, percentPertub=0):
    
    data = getData_fromFileUser(userfile, datatype=datatype, useTempInfo=useTempInfo, return_timestamp=False)
    if percentPertub>0: 
        idx_samples_pertub = np.random.rand(data.shape[0])<=percentPertub
        # min_data = np.min(data,axis=0)
        # max_data = np.max(data,axis=0)
        dataPertub = np.array(np.random.uniform(low=min_data, high=max_data, size=[sum(idx_samples_pertub),data.shape[1]]), dtype='int')
        
        data[idx_samples_pertub,:]= dataPertub

        
    data = scaler.transform(data)

    return tf.data.Dataset.from_tensor_slices((data, data)).batch(20)
    

all_train_data = [getData_fromFileUser(users[iuser], datatype=datatype, useTempInfo=useTempInfo, return_timestamp=False ) for iuser in range(0,len(users))]





for _ in range(0,20):
    usersKnown = groupUsers
    
    scores_foreachUser=[]
    
    
    ## Perturb Data users 
    users_pertub = np.random.choice(usersKnown, numClientDenial, replace=False)
    print('Usuarios perturbados')
    print(users_pertub)
    list_users_pertub = [] 
    for user in usersKnown :
        if user in users_pertub:
            list_users_pertub.append(percent)
        else:
            list_users_pertub.append(0)


    # Get train data ONLY for user known
    userKnown_train_data = [all_train_data[numIDuser.index(IDUser)] for IDUser in usersKnown]
    
    
    # Scaler
    scalerGlobal = MinMaxScaler()
    userKnown_train_data_scaled = scalerGlobal.fit_transform(np.concatenate(userKnown_train_data))
    
    X_train = [client_data(users[numIDuser.index(user)],datatype=datatype, useTempInfo=useTempInfo, scaler=scalerGlobal, percentPertub=list_users_pertub[iu]) for iu,user in enumerate(usersKnown)]
    
    
    
    # scaler = MinMaxScaler()
    # X_train = scaler.fit_transform(X_train)
        
    ### Model Train
    
    num_feats = X_train[0].element_spec[0].shape[1]

    def tff_model():
        
        if modelName == 'Autoencoder_1_16': 
            autoencoder =  models.autoencoder_skeleton_1D(num_feats,16)
        elif modelName == 'Autoencoder_1_8': 
            autoencoder = models.autoencoder_skeleton_1D(num_feats,8)
        elif modelName == 'Autoencoder_1_4': 
            autoencoder = models.autoencoder_skeleton_1D(num_feats,4)
        elif modelName == 'Autoencoder_2_16_4': 
            autoencoder = models.autoencoder_skeleton_2D(num_feats,16,4)
        elif modelName == 'Autoencoder_2_16_8': 
            autoencoder = models.autoencoder_skeleton_2D(num_feats,16,8)
        elif modelName == 'Autoencoder_2_8_4': 
            autoencoder = models.autoencoder_skeleton_2D(num_feats,8,4)
                
        elif modelName == 'Variational_Autoencoder_1_16': 
            autoencoder,loss =  models.variational_autoencoder_skeleton_1D(num_feats,16)
        elif modelName == 'Variational_Autoencoder_1_8': 
            autoencoder,loss = models.variational_autoencoder_skeleton_1D(num_feats,8)
        elif modelName == 'Variational_Autoencoder_1_4': 
            autoencoder,loss = models.variational_autoencoder_skeleton_1D(num_feats,4)
        elif modelName == 'Variational_Autoencoder_2_16_4': 
            autoencoder,loss = models.variational_autoencoder_skeleton_2D(num_feats,16,4)
        elif modelName == 'Variational_Autoencoder_2_16_8': 
            autoencoder,loss = models.variational_autoencoder_skeleton_2D(num_feats,16,8)
        elif modelName == 'Variational_Autoencoder_2_8_4': 
            autoencoder,loss = models.variational_autoencoder_skeleton_2D(num_feats,8,4)
            
        else:
            print("No es correcto el modelo")
            
        return tff.learning.from_keras_model(
            autoencoder,
            input_spec=X_train[0].element_spec,
            loss=tf.keras.losses.MeanSquaredError())
    
    
    if algorithmAgregation == 'mean':
        aggregation_factory = tff.aggregators.MeanFactory()
    elif algorithmAgregation == 'median':
        median = tff.aggregators.PrivateQuantileEstimationProcess.no_noise(
            initial_estimate=1.0, target_quantile=0.5, learning_rate=0.2)
        
        aggregation_factory = tff.aggregators.zeroing_factory(
            zeroing_norm=median,
            inner_agg_factory=tff.aggregators.MeanFactory()) 
            
    elif algorithmAgregation == 'trimmed_80':
        zeroing_norm = tff.aggregators.PrivateQuantileEstimationProcess.no_noise(
            initial_estimate=10.0,
            target_quantile=0.80,
            learning_rate=math.log(10),
            multiplier=2.0,
            increment=1.0)
        aggregation_factory = tff.aggregators.zeroing_factory(
            zeroing_norm=zeroing_norm,
            inner_agg_factory=tff.aggregators.MeanFactory())
    
    elif algorithmAgregation == 'trimmed_60':
        zeroing_norm = tff.aggregators.PrivateQuantileEstimationProcess.no_noise(
            initial_estimate=10.0,
            target_quantile=0.60,
            learning_rate=math.log(10),
            multiplier=2.0,
            increment=1.0)
        aggregation_factory = tff.aggregators.zeroing_factory(
            zeroing_norm=zeroing_norm,
            inner_agg_factory=tff.aggregators.MeanFactory())


    elif algorithmAgregation == 'DiferencialPrivacy':
        aggregation_factory = tff.learning.model_update_aggregator.dp_aggregator(
            0.2, len(usersKnown))
    elif algorithmAgregation == 'trimmed_80_l2':
        clipping_norm = tff.aggregators.PrivateQuantileEstimationProcess.no_noise(
            initial_estimate=1.0,
            target_quantile=0.80,
            learning_rate=0.2)
        aggregation_factory = tff.aggregators.clipping_factory(
            clipping_norm=clipping_norm,
            inner_agg_factory=tff.aggregators.MeanFactory())
    
    elif algorithmAgregation == 'trimmed_60_l2':
        clipping_norm = tff.aggregators.PrivateQuantileEstimationProcess.no_noise(
            initial_estimate=1.0,
            target_quantile=0.60,
            learning_rate=0.2)
        aggregation_factory = tff.aggregators.clipping_factory(
            clipping_norm=clipping_norm,
            inner_agg_factory=tff.aggregators.MeanFactory())
        
    else :
        print("not trainer selected")
        exit()
                    
    
    
    trainer = tff.learning.algorithms.build_weighted_fed_avg(
        tff_model,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0),
        model_aggregator=aggregation_factory)
        
        
        

                    
    
            
    state = trainer.initialize()
    
    min_loss = 1000000
    steps_withouReduceloss=0
    steps_max_training=200
    for iter_train in range(steps_max_training):
        result = trainer.next(state, X_train)
        state = result.state
        metrics = result.metrics
        print(f" iter:{iter_train}/{steps_max_training} - loss:{metrics['client_work']['train']['loss']}")
        
        if iter_train==0:
            min_loss=metrics['client_work']['train']['loss']
            state_minloss=state
            
        if (metrics['client_work']['train']['loss']+1e-5)<min_loss:
            min_loss = metrics['client_work']['train']['loss']
            steps_withouReduceloss=0
            state_minloss=state
        else:
            steps_withouReduceloss+=1
        
        if steps_withouReduceloss==5:
            print('Stopping the training')
            state=state_minloss
            break
        
    if modelName == 'Autoencoder_1_16': 
        autoencoder =  models.autoencoder_skeleton_1D(num_feats,16)
    elif modelName == 'Autoencoder_1_8': 
        autoencoder = models.autoencoder_skeleton_1D(num_feats,8)
    elif modelName == 'Autoencoder_1_4': 
        autoencoder = models.autoencoder_skeleton_1D(num_feats,4)
    elif modelName == 'Autoencoder_2_16_4': 
        autoencoder = models.autoencoder_skeleton_2D(num_feats,16,4)
    elif modelName == 'Autoencoder_2_16_8': 
        autoencoder = models.autoencoder_skeleton_2D(num_feats,16,8)
    elif modelName == 'Autoencoder_2_8_4': 
        autoencoder = models.autoencoder_skeleton_2D(num_feats,8,4)
    
            
    elif modelName == 'Variational_Autoencoder_1_16': 
        autoencoder,loss =  models.variational_autoencoder_skeleton_1D(num_feats,16)
    elif modelName == 'Variational_Autoencoder_1_8': 
        autoencoder,loss = models.variational_autoencoder_skeleton_1D(num_feats,8)
    elif modelName == 'Variational_Autoencoder_1_4': 
        autoencoder,loss = models.variational_autoencoder_skeleton_1D(num_feats,4)
    elif modelName == 'Variational_Autoencoder_2_16_4': 
        autoencoder,loss = models.variational_autoencoder_skeleton_2D(num_feats,16,4)
    elif modelName == 'Variational_Autoencoder_2_16_8': 
        autoencoder,loss = models.variational_autoencoder_skeleton_2D(num_feats,16,8)
    elif modelName == 'Variational_Autoencoder_2_8_4': 
        autoencoder,loss = models.variational_autoencoder_skeleton_2D(num_feats,8,4)
        
        
    else:
        print("No es correcto el modelo")
    autoencoder.compile(
            loss=tf.keras.losses.MeanSquaredError)
    
    model_weights = trainer.get_model_weights(state)
    
    model_weights.assign_weights_to(autoencoder)
    
    
    X = [np.concatenate(list(X_train[i].as_numpy_iterator()),axis=1)[0,:,:]   for i in range(len(usersKnown))]
    X = np.concatenate(X)  
        
    reconstructions = autoencoder.predict(X)
    train_loss = tf.keras.losses.mse(reconstructions, X)
        
    threshold = np.mean(train_loss)+np.std(train_loss)
        
    ### Global Model Evaluate

    X_test = []
    X_test_pertub = []
    Y_test = []
    for kuser in range(0,len(users)):
        filetest = users[kuser].replace('train','test')
        numIDuser_kuser = int(filetest.split('user')[1].split('_')[0])
        if numIDuser_kuser in groupUsers:
            label=0
        elif numIDuser_kuser in impostorUsers:
            label=1 #Anomaly
            
        else:
            continue
            
        if verbose:
            print(" USER {}-{}: {}".format(kuser, len(users),filetest))
				    
        X_test_user = getData_fromFileUser(filetest, datatype=datatype, useTempInfo=useTempInfo, return_timestamp=False)
        
 
        X_test_user = scalerGlobal.transform(X_test_user)
        
        X_test.append(X_test_user)
        Y_test.append(np.ones(X_test_user.shape[0])*label)
        
        if numIDuser_kuser in users_pertub:
            X_test_pertub.append(X_test_user)
    
    X_test = np.concatenate(X_test,axis=0)
    Y_test = np.concatenate(Y_test,axis=0)
    

    
    reconstructions = autoencoder.predict(X_test)
    test_loss = tf.keras.losses.mse(reconstructions, X_test).numpy()
    
    Y_pred = test_loss>threshold
    

    auc_sc = accuracy_score(Y_test, Y_pred)
    precision_sc = precision_score(Y_test, Y_pred)
    recall_sc = recall_score(Y_test, Y_pred)
    f1_score_sc = f1_score(Y_test, Y_pred)
    
    specificity_sc = sum( Y_pred[Y_test==0]==False)/sum(Y_test==0)
    
    auc.append(auc_sc)
    precision.append(precision_sc)
    recall.append(recall_sc)
    F1_score.append(f1_score_sc)
    
    specificity.append(specificity_sc)
    
    tnr_sc=0.0
    if len(X_test_pertub)>0:
        
        X_test_pertub = np.concatenate(X_test_pertub, axis=0)
        reconstructions = autoencoder.predict(X_test_pertub)
        test_pertub_loss = tf.keras.losses.mse(reconstructions, X_test_pertub).numpy()
        
        Y_pertub_pred = test_pertub_loss>threshold    
        
        tnr_sc = sum(Y_pertub_pred==0)/len(Y_pertub_pred)
        # print(f"{i} - {len(Y_pertub_pred)} - {sum(Y_pertub_pred==0)} - {tnr_sc} ")
        tnr.append(tnr_sc)
    
    string = "{} - {} - {} - {} - {} - {} - {} ".format(users_pertub, auc_sc, precision_sc, recall_sc, f1_score_sc, specificity_sc, tnr_sc) 
    print(string)
    a_file.write(string)    
        
        
num_test=len(auc)
string = "{} - ( {} , {} ) - ( {} , {} ) - ( {} , {} ) - ( {} , {} ) - ( {} , {} ) - ( {} , {} )".format(users_pertub,
    100*(np.mean(auc)-1.96*np.std(auc)/np.sqrt(num_test)),100*( np.mean(auc)+1.96*np.std(auc)/np.sqrt(num_test)),
    100*(np.mean(precision)-1.96*np.std(precision)/np.sqrt(num_test)),100*( np.mean(precision)+1.96*np.std(precision)/np.sqrt(num_test)),
    100*(np.mean(recall)-1.96*np.std(recall)/np.sqrt(num_test)),100*( np.mean(recall)+1.96*np.std(recall)/np.sqrt(num_test)), 
    100*(np.mean(F1_score)-1.96*np.std(F1_score)/np.sqrt(num_test)),100*( np.mean(F1_score)+1.96*np.std(F1_score)/np.sqrt(num_test)), 
    100*(np.mean(specificity)-1.96*np.std(specificity)/np.sqrt(num_test)),100*( np.mean(specificity)+1.96*np.std(specificity)/np.sqrt(num_test)), 
    100*(np.mean(tnr)-1.96*np.std(tnr)/np.sqrt(num_test)),100*( np.mean(tnr)+1.96*np.std(tnr)/np.sqrt(num_test)))

print(string)
a_file.write(string)

a_file.close()















