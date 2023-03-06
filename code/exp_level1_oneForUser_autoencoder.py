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

# import nest_asyncio
# nest_asyncio.apply()



random.seed(21712)
np.random.seed(21712)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('datatype', type=str, default='sensors', help='sensors, statistics or voice')
parser.add_argument('model', type=str)
parser.add_argument('--typeCom', type=str, default=False, help='Verbose')
parser.add_argument('--useTempInfo', type=bool, default=False, help='Verbose')
parser.add_argument('--verbose', type=bool, default=False, help='Verbose')
parser.add_argument('--typeScale', type=str, default='global', help='Verbose')



args = parser.parse_args()
print(args)
datatype = args.datatype 
modelName = args.model
verbose = args.verbose
typeCom = args.typeCom
useTempInfo = args.useTempInfo
typeScale = args.typeScale

#datatype = 'statistics'
#modelName = 'Autoencoder_1_16'
#verbose = True
#typeCom='random'

#pathToFiles='../data/14days/'
#useTempInfo=False

pathToFiles='data/14days/'


nameModel=modelName #str(model).split('.')[-1].replace("'","").replace('>','')


pathToScores='./scores/experiment_1/SecuriteLevel_1/'+nameModel+'_' 
pathToScores+= 'scaler'+typeScale+'_'
pathToScores+= typeCom
pathToScores+='_tempInfo' if useTempInfo else '_noTempInfo'

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

        

# Number of different combinations or max 20. 
num_test=20 


    
auc=[]
precision=[]
recall=[]
F1_score=[]



# Get train data of ALL users
all_train_data = [getData_fromFileUser(users[iuser], datatype=datatype, useTempInfo=useTempInfo, return_timestamp=False ) for iuser in range(0,len(users))]

# Get individual scaler for ALL users
if typeScale=='user' or typeScale=='model':
    all_scalers = []
    all_train_data_scaler = []
    for iuser in range(0,len(users)):
        scaler = MinMaxScaler()
        all_train_data_scaler.append(scaler.fit_transform(all_train_data[iuser]))
        all_scalers.append(scaler)

for iter in progressbar(range(0,num_test)):
    usersKnown = groupUsers

    if typeScale=='global':
        userKnown_train_data = [all_train_data[numIDuser.index(IDUser)] for IDUser in usersKnown]
        
        # Scaler
        scalerGlobal = MinMaxScaler()
        userKnown_train_data_scaled = scalerGlobal.fit_transform(np.concatenate(userKnown_train_data))
    
    
    scores_foreachUser=[]
    
    models_users = []
    thresholds = []
    for iuser in usersKnown: 
        
        print(f" Training Model {iuser}")
        
        if typeScale=='global':
            scaler_user=scalerGlobal
        else: 
            scaler_user = all_scalers[numIDuser.index(iuser)]
        
        user_train_data_scaled = scaler_user.transform(getData_fromFileUser(users[numIDuser.index(iuser)], datatype=datatype, useTempInfo=useTempInfo, return_timestamp=False ))
       
        
        X_train = user_train_data_scaled
        ### Model Train
        
        num_feats = X_train.shape[1]
    
    
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
        
        
        if not 'Variational' in modelName: 
            autoencoder.compile(optimizer='adam', loss='mae')
        else: 
            autoencoder.compile(optimizer='adam', loss=loss)
        
        
        history = autoencoder.fit(X_train, X_train, 
              epochs=1000, 
              validation_split=0.2,
              batch_size=512,
              shuffle=True, 
              verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(
                            monitor='val_loss',
                            min_delta=0,
                            patience=10,
                            verbose=1,
                            mode='auto',
                            baseline=None,
                            restore_best_weights=True
                            )]
            )
    
        reconstructions = autoencoder.predict(X_train)
        train_loss = tf.keras.losses.mae(reconstructions, X_train)
        
            
        threshold = np.mean(train_loss)+np.std(train_loss)
        
        models_users.append(autoencoder)
        thresholds.append(threshold)
        
    ### Model Evaluate

    X_test = []
    Y_test = []
    for kuser in range(0,len(users)):
        filetest = users[kuser].replace('train','test')
        numIDuser_kuser = int(filetest.split('user')[1].split('_')[0])
        if numIDuser_kuser in usersKnown:
            label=0
        elif numIDuser_kuser in impostorUsers:
            label=1 #Anomaly
        
        else: 
            continue
        
        
        if verbose:
            print(" USER {}-{}: {}".format(kuser, len(users),filetest))
				
        X_test_user = getData_fromFileUser(filetest, datatype=datatype, useTempInfo=useTempInfo, return_timestamp=False)
        
        if typeScale=='user':
            scaler = all_scalers[kuser]
            X_test_user = scaler.transform(X_test_user)
        
        X_test.append(X_test_user)
        Y_test.append(np.ones(X_test_user.shape[0])*label)
    
    X_test = np.concatenate(X_test,axis=0)
    Y_test = np.concatenate(Y_test,axis=0)
    
    Y_pred_users=[]
    for iuser in range(0,len(usersKnown)): 
        
        iuser_ID = usersKnown[iuser]
        
        if typeScale=='model':
            scaler = all_scalers[numIDuser.index(iuser_ID)]
            X_test = scaler.transform(X_test)
            
        if typeScale=='global':
            X_test = scalerGlobal.transform(X_test)
        
        reconstructions = models_users[iuser].predict(X_test)
        test_loss = tf.keras.losses.mae(reconstructions, X_test).numpy()
    
    
        Y_pred_users.append(test_loss>thresholds[iuser])
    
    Y_pred_users = np.array(Y_pred_users)
    Y_pred_users_negado = ~ Y_pred_users
    
    Y_pred_negado = np.any(Y_pred_users_negado,axis=0)
    
    Y_pred = ~Y_pred_negado

    auc_sc = accuracy_score(Y_test, Y_pred)
    precision_sc = precision_score(Y_test, Y_pred)
    recall_sc = recall_score(Y_test, Y_pred)
    f1_score_sc = f1_score(Y_test, Y_pred)
    
    if verbose:
        print(str(usersKnown))
        print(thresholds)
        print(str(auc_sc) +','+str(precision_sc) + ',' + str(recall_sc)+ ',' + str(f1_score_sc) +"\n")
    a_file.write(str(usersKnown))
    a_file.write(str(auc_sc) +','+str(precision_sc) + ',' + str(recall_sc)+ ',' + str(f1_score_sc) +"\n")
    
    auc.append(auc_sc)
    precision.append(precision_sc)
    recall.append(recall_sc)
    F1_score.append(f1_score_sc)
    
string = "( {} , {} ) - ( {} , {} ) - ( {} , {} ) - ( {} , {} )".format(
    100*(np.mean(auc)-1.96*np.std(auc)/np.sqrt(num_test)),100*( np.mean(auc)+1.96*np.std(auc)/np.sqrt(num_test)),
    100*(np.mean(precision)-1.96*np.std(precision)/np.sqrt(num_test)),100*( np.mean(precision)+1.96*np.std(precision)/np.sqrt(num_test)),
    100*(np.mean(recall)-1.96*np.std(recall)/np.sqrt(num_test)),100*( np.mean(recall)+1.96*np.std(recall)/np.sqrt(num_test)), 
    100*(np.mean(F1_score)-1.96*np.std(F1_score)/np.sqrt(num_test)),100*( np.mean(F1_score)+1.96*np.std(F1_score)/np.sqrt(num_test)))

print(string)
a_file.write(string)

a_file.close()
