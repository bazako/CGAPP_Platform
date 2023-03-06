# -*- coding: utf-8 -*-

import glob
import os 
import numpy as np
from progressbar import progressbar
import argparse
import tensorflow as tf
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler

from getData import getData_fromFileUser
from getUserCombinations import getUserCombinations

import models 







random.seed(21712)
np.random.seed(21712)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('datatype', type=str, default='sensors', help='sensors, statistics or voice')
parser.add_argument('model', type=str)
parser.add_argument('--typeCom', type=str, default="Workers", help='Verbose')
parser.add_argument('--useTempInfo', type=bool, default=False, help='Verbose')
parser.add_argument('--verbose', type=bool, default=False, help='Verbose')
parser.add_argument('--individualScaler', type=bool, default=False, help='Verbose')


args = parser.parse_args()
print(args)
datatype = args.datatype 
modelName = args.model
verbose = args.verbose
typeCom = args.typeCom
useTempInfo = args.useTempInfo
individualScaler = args.individualScaler

#datatype = 'statistics'
#modelName = 'Autoencoder_1_16'
#verbose = True
#fixUsers=True
#num_feats=40
#typeCom = 'random'
#pathToFiles='../data/14days/'
#useTempInfo=False

pathToFiles='data/14days/'


nameModel=modelName #str(model).split('.')[-1].replace("'","").replace('>','')

pathToScores='./scores/experiment_1/SecuriteLevel_0/'+nameModel+'_' 
pathToScores+= 'scalerIndividual_' if individualScaler else 'scalerGlobal_'
pathToScores+= typeCom
pathToScores+='_tempInfo' if useTempInfo else '_noTempInfo'


users=glob.glob(pathToFiles+'train/*'+datatype+'.txt')
users.sort()



# Get ID users
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
        userKnown_train_data_scaled = [all_train_data_scaler[numIDuser.index(IDUser)] for IDUser in usersKnown]
        userKnown_train_data_scaled = np.concatenate(userKnown_train_data_scaled)
    
    else: 
        # Get train data ONLY for user known
        userKnown_train_data = [all_train_data[numIDuser.index(IDUser)] for IDUser in usersKnown]
        
        # Scaler
        scalerGlobal = MinMaxScaler()
        userKnown_train_data_scaled = scalerGlobal.fit_transform(np.concatenate(userKnown_train_data))
    
        
        
    
    X_train = userKnown_train_data_scaled

    
    
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
        autoencoder.compile(optimizer='adam', loss='mse')
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
    train_loss = tf.keras.losses.mse(reconstructions, X_train)
    
        
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
    

    
    reconstructions = autoencoder.predict(X_test)
    test_loss = tf.keras.losses.mse(reconstructions, X_test).numpy()
    
    
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
