#!bin/bash

# Autoencoders 1-16
python code/exp_level0_oneModel_autoencoder.py statistics Autoencoder_1_16 --typeCom Workers

# Autoencoder 2-16_8
python code/exp_level0_oneModel_autoencoder.py statistics Autoencoder_2_16_8 --typeCom Workers

# Autoender 1-16
python code/exp_level0_oneModel_autoencoder.py statistics Variational_Autoencoder_1_16 --typeCom Workers

# Variational Autoencoder 2-16_8
python code/exp_level0_oneModel_autoencoder.py statistics Variational_Autoencoder_2_16_8 --typeCom Workers

# KNN-50
python code/exp_level0_oneModel_pyOD.py statistics KNN --config n_neighbors:50 --typeCom Workers 

