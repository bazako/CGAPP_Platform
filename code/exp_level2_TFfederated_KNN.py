import glob
import pandas as pd
import os 
import numpy as np

from getData import getData_fromFileUser
from getUserCombinations import getUserCombinations
# import json
from progressbar import progressbar
import argparse
# from utils import calculate_eer, calculate_metrics
import models 
from itertools import combinations
import random

from sklearn.preprocessing import StandardScaler,MinMaxScaler

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras import losses

import tensorflow as tf
import tensorflow_federated as tff

# import nest_asyncio
# nest_asyncio.apply()



random.seed(21712)
np.random.seed(21712)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('datatype', type=str, default='sensors', help='sensors, statistics or voice')
parser.add_argument('model', type=str)
parser.add_argument('--typeCom', type=str, default="Workers", help='Verbose')
parser.add_argument('--useTempInfo', type=bool, default=False, help='Verbose')
parser.add_argument('--verbose', type=bool, default=False, help='Verbose')
parser.add_argument('--iter', type=int, default=10, help='Verbose')

parser.add_argument('--individualScaler', type=bool, default=False, help='Verbose')


args = parser.parse_args()
print(args)
datatype = args.datatype 
modelName = args.model
verbose = args.verbose
typeCom = args.typeCom
useTempInfo = args.useTempInfo
individualScaler = args.individualScaler
maxiter = args.iter

#datatype = 'statistics'
#modelName = 'Autoencoder_1_16'
#verbose = True
#fixUsers=False
#num_feats=40
#pathToFiles='../data/14days/'
#useTempInfo=False

pathToFiles='data/14days/'


nameModel=modelName #str(model).split('.')[-1].replace("'","").replace('>','')

pathToScores='./scores/experiment_1/SecuriteLevel_2/KNN_' 
pathToScores+='Scaler' + 'Individual' if individualScaler else 'Global'
pathToScores+='_'+typeCom+'_noTempInfo_iter'+str(maxiter)


if useTempInfo:
    pathToScores=pathToScores.replace('noTempInfo','yesTempInfo')


users=glob.glob(pathToFiles+'train/*'+datatype+'.txt')
users.sort()


# Users in groups of 10 persons
numIDuser = []
for kuser in range(0,len(users)):
    filetest = users[kuser]
    numIDuser.append(int(filetest.split('user')[1].split('_')[0]))

groupUsers, impostorUsers, middleUsers = getUserCombinations( numIDuser, userCombination=typeCom ) 
        
os.makedirs(pathToScores,exist_ok=True)   
file_resume = pathToScores + '/resume_'+datatype+'.txt'
a_file=open(file_resume, "a")

        

num_test=20 


    
auc=[]
precision=[]
recall=[]
F1_score=[]



def client_data(userfile,datatype, useTempInfo, scaler):
    
    data = getData_fromFileUser(userfile, datatype=datatype, useTempInfo=useTempInfo, return_timestamp=False)
    data = data.astype(dtype=np.float32)
    data = scaler.transform(data)
    return tf.data.Dataset.from_tensor_slices(data)
    

all_train_data = [getData_fromFileUser(users[iuser], datatype=datatype, useTempInfo=useTempInfo, return_timestamp=False ) for iuser in range(0,len(users))]

# Get individual scaler for ALL users
if individualScaler:
    all_scalers = []
    all_train_data_scaler = []
    for iuser in range(0,len(users)):
        scaler = MinMaxScaler()
        all_train_data_scaler.append(scaler.fit_transform(all_train_data[iuser]))
        all_scalers.append(scaler)



for iter in progressbar(range(0,num_test)):
    usersKnown = groupUsers
    
    scores_foreachUser=[]
       
    if individualScaler: 
        X_train = [client_data(users[numIDuser.index(iuser)],datatype=datatype, useTempInfo=useTempInfo, scaler=all_scalers[numIDuser.index(iuser)]) for iuser in usersKnown]
    else: 
        # Get train data ONLY for user known
        userKnown_train_data = [all_train_data[numIDuser.index(IDUser)] for IDUser in usersKnown]
        
        # Scaler
        scalerGlobal = MinMaxScaler()
        userKnown_train_data_scaled = scalerGlobal.fit_transform(np.concatenate(userKnown_train_data))
        
        X_train = [client_data(users[numIDuser.index(iuser)],datatype=datatype, useTempInfo=useTempInfo, scaler=scalerGlobal) for iuser in usersKnown]
       
        
    ### Model Train
    
    num_feats = X_train[0].element_spec.shape[0]

  
    dataShape=(num_feats,)
    

    trainer = tff.learning.algorithms.build_fed_kmeans(int(nameModel),dataShape)
                    
                    
    
            
    state = trainer.initialize()
    
    min_loss = 1000000
    steps_withouReduceloss=0
    steps_max_training=maxiter
    for iter_train in range(steps_max_training):
        result = trainer.next(state, X_train)
        state = result.state
        metrics = result.metrics
        print(f" iter:{iter_train}/{steps_max_training}")


    
    model_weights = trainer.get_model_weights(state)
    


    X = [np.array(list(X_train[i].as_numpy_iterator()) )   for i in range(len(usersKnown))]
    X = np.concatenate(X)  

    from scipy.spatial.distance import cdist
    
    dm = cdist(X, model_weights,'minkowski', p=2.)
        
    train_loss = np.max(dm,axis=1)
        
    threshold = np.mean(train_loss)+np.std(train_loss)
        
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
        
        
        if individualScaler: 
            X_test_user = all_scalers[kuser].transform(X_test_user)
        else: 
            X_test_user = scalerGlobal.transform(X_test_user)
        
        X_test.append(X_test_user)
        Y_test.append(np.ones(X_test_user.shape[0])*label)
    
    X_test = np.concatenate(X_test,axis=0)
    Y_test = np.concatenate(Y_test,axis=0)
    

    
    dm_test = cdist(X_test, model_weights,'minkowski', p=2.)
    
    test_loss = np.max(dm_test,axis=1)
    # for i in range(500,1000):
    #     threshold = 6*i/1000
    Y_pred = test_loss>threshold
    

    auc_sc = accuracy_score(Y_test, Y_pred)
    precision_sc = precision_score(Y_test, Y_pred)
    recall_sc = recall_score(Y_test, Y_pred)
    f1_score_sc = f1_score(Y_test, Y_pred)
    
    if verbose:
        print(str(usersKnown))
        print(threshold)
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
