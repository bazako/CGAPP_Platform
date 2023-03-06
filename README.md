# CGAPP_Platform
CGAPP platform code for results to Industry 4.0


The dataset is availabe in https://figshare.com/articles/dataset/S3Dataset_zip/14410229/2


## Requirements
A requirements.txt file is attached to be able to create the environment in conda directly with the following command:

`$ conda create --name CGAPP_Platform --file requirements.txt `


## Initial Steps

To get the Dataset, execute the follow comands:

`$ wget https://figshare.com/ndownloader/files/27546551/S3Dataset.zip`

`$ unzip S3Dataset.zip`

Activate your enviroment : `$ conda activate CGAPP_Platform`

`$ python code/restructureDataFolderByUser.py`

## Run Experiment 1

`$ bash run_experiment1.sh`

Note: the first part of the experiment (selections of the best params) is commented, due to this stage is very heavy in time of computation.

## Run Experiment 2

`$ bash run_experiment2.sh`
