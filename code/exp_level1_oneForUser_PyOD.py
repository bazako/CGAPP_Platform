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
import random

from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# import nest_asyncio
# nest_asyncio.apply()



random.seed(21712)
np.random.seed(21712)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('datatype', type=str, default='sensors', help='sensors, statistics or voice')
parser.add_argument('model', type=str)
parser.add_argument('--config', type=str, default="", help='Verbose')
parser.add_argument('--typeCom', type=str, default=False, help='Verbose')
parser.add_argument('--useTempInfo', type=bool, default=False, help='Verbose')
parser.add_argument('--verbose', type=bool, default=False, help='Verbose')
parser.add_argument('--typeScale', type=str, default='global', help='Verbose')



args = parser.parse_args()
print(args)
datatype = args.datatype 
modelName = args.model
configModel = args.config
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




users=glob.glob(pathToFiles+'train/*'+datatype+'.txt')
users.sort()



# Users in groups of 10 persons
numIDuser = []
for kuser in range(0,len(users)):
    filetest = users[kuser]
    numIDuser.append(int(filetest.split('user')[1].split('_')[0]))

groupUsers, impostorUsers, middleUsers = getUserCombinations( numIDuser, userCombination=typeCom) 



        

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

if modelName == 'ABOD':
    parameters = {'n_neighbors':[5,10,20,30,50]}
if modelName == 'OCSVM':
	parameters = {'kernel':('linear', 'rbf', 'sigmoid','poly'),'nu':[0.1,0.2,0.5,0.7]}
if modelName == 'KNN':
	parameters = {'n_neighbors':[1,5,10,25,50,75,100,150,200]}
if modelName == 'IF':
	parameters = {'n_estimators':[1,5,10,50,100], 'max_samples':[0.2,0.5,0.7,1.0], 'max_features':[0.2,0.5,0.7,1.0]}


if not configModel == "":
    parameters={}
    for configString in configModel.split(','):
        if configString.split(':')[0] == 'n_neighbors' or configString.split(':')[0] == 'n_estimators' : 
            parameters[configString.split(':')[0]] = int(configString.split(':')[1])
        elif configString.split(':')[0] == 'nu' or configString.split(':')[0] == 'max_samples' or configString.split(':')[0] == 'max_features' : 
            parameters[n_neighbors] = float(configString.split(':')[1])
        else:
            parameters[n_neighbors] = float(configString.split(':')[1])

    experiments = [parameters]

else: 

    from sklearn.model_selection import ParameterGrid
    experiments =  list(ParameterGrid(parameters))

print(experiments)


for i_exp,params in enumerate(experiments):
    print(f"Experiment {i_exp}: {params}")
    str_params = str(params).replace("'","").replace(':','_').replace(',','_').replace('.','').replace(' ','').replace('{','').replace('}','')
    
    pathToScores='./scores/experiment_1/SecuriteLevel_1/'+nameModel+'_' 
    pathToScores+= 'scaler' + typeScale +'_'
    pathToScores+= typeCom
    pathToScores+='_tempInfo' if useTempInfo else '_noTempInfo'
    pathToScores+= '_'+str_params+'/'
    
    os.makedirs(pathToScores,exist_ok=True)   
    file_resume = pathToScores + '/resume_'+datatype+'.txt'
    a_file=open(file_resume, "a")
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
        
        
            if modelName == 'ABOD': 
                from pyod.models.abod import ABOD
                clf =  ABOD()
            elif modelName == 'KNN': 
                from pyod.models.knn import KNN
                clf = KNN()
            elif modelName == 'IF': 
                from pyod.models.iforest import IForest
                clf = IForest()
            elif modelName == 'OCSVM': 
                from pyod.models.ocsvm import OCSVM
                clf = OCSVM()
                
            else:
                print("No es correcto el modelo")
                    
            clf.fit(X_train)
            
            y_train_scores = clf.predict(X_train)
            
                
            threshold = np.mean(y_train_scores)+np.std(y_train_scores)
            
            models_users.append(clf)
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
            
            y_scores = models_users[iuser].predict(X_test)
            
        
            Y_pred_users.append(y_scores>thresholds[iuser])
        
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
