# -*- coding: utf-8 -*-

import glob
import os 
import numpy as np
from progressbar import progressbar
import argparse
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler

from getData import getData_fromFileUser
from getUserCombinations import getUserCombinations


random.seed(21712)
np.random.seed(21712)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('datatype', type=str, default='sensors', help='sensors, statistics or voice')
parser.add_argument('model', type=str)
parser.add_argument('--config', type=str, default="", help='Verbose')
parser.add_argument('--typeCom', type=str, default="Workers", help='Verbose')
parser.add_argument('--useTempInfo', type=bool, default=False, help='Verbose')
parser.add_argument('--verbose', type=bool, default=False, help='Verbose')
parser.add_argument('--individualScaler', type=bool, default=False, help='Verbose')


args = parser.parse_args()
print(args)
datatype = args.datatype 
modelName = args.model
configModel = args.config
verbose = args.verbose
typeCom = args.typeCom
useTempInfo = args.useTempInfo
individualScaler = args.individualScaler

#datatype = 'statistics'
#modelName = 'Autoencoder_1_16'
#verbose = True
#fixUsers=True
#num_feats=40
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



        

# Number of different combinations or max 20. 
num_test=10

if typeCom=='cluster':
    num_test=1

# Get 20 randoms combinations


    
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


if modelName == 'ABOD':
    parameters = {'n_neighbors':[1,2,5,10,20,30,50]}
if modelName == 'OCSVM':
	parameters = {'kernel':('linear', 'rbf', 'sigmoid','poly'),'nu':[0.1,0.2,0.5,0.7]}
if modelName == 'KNN':
	parameters = {'n_neighbors':[1,5,10,20,30,50,100,200,300,500,1000]}
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
    
    pathToScores='./scores/experiment_1/SecuriteLevel_0/'+nameModel+'_' 
    pathToScores+= 'scalerIndividual_' if individualScaler else 'scalerGlobal_'
    pathToScores+= typeCom
    pathToScores+='_tempInfo' if useTempInfo else '_noTempInfo'
    pathToScores+='_'+str_params+'/'
    
    os.makedirs(pathToScores,exist_ok=True)   
    file_resume = pathToScores + '/resume_'+datatype+'.txt'
    a_file=open(file_resume, "a")

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
    
    
        if modelName == 'ABOD': 
            from pyod.models.abod import ABOD
            clf =  ABOD(**params)
        elif modelName == 'KNN': 
            from pyod.models.knn import KNN
            clf = KNN(**params)
        elif modelName == 'IF': 
            from pyod.models.iforest import IForest
            clf = IForest(**params)
        elif modelName == 'OCSVM': 
            from pyod.models.ocsvm import OCSVM
            clf = OCSVM(**params)
            
        else:
            print("No es correcto el modelo")
                
        clf.fit(X_train)
        
        y_train_scores = clf.predict(X_train)
       
            
        threshold = np.mean(y_train_scores)+np.std(y_train_scores)
            
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
        
    
        
        y_scores = clf.predict(X_test)
        
        Y_pred = y_scores>threshold
        
    
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
