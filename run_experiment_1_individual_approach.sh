#!bin/bash

# Autoencoders 1-16
python code/exp_level1_oneForUser_autoencoder.py statistics Autoencoder_1_16 --typeCom Workers --typeScale user	

# Autoencoder 2-16_8
python code/exp_level1_oneForUser_autoencoder.py statistics Autoencoder_2_16_8 --typeCom Workers --typeScale user	

# Autoender 1-16
python code/exp_level1_oneForUser_autoencoder.py statistics Variational_Autoencoder_1_16 --typeCom Workers --typeScale user	

# Variational Autoencoder 2-16_8
python code/exp_level1_oneForUser_autoencoder.py statistics Variational_Autoencoder_2_16_8 --typeCom Workers --typeScale user	

# KNN-50
python code/exp_level1_oneForUser_PyOD.py statistics KNN --config n_neighbors:50 --typeCom Workers --typeScale user	