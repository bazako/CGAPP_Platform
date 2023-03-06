#!bin/bash



# Autoencoders 1-16
python code/exp_level2_TFfederated_autoencoder.py statistics Autoencoder_1_16 --iter 200 --typeCom Workers

# Autoencoder 2-16_8
python code/exp_level2_TFfederated_autoencoder.py statistics Autoencoder_2_16_8 --iter 200 --typeCom Workers

# Autoender 1-16
python code/exp_level2_TFfederated_autoencoder.py statistics Variational_Autoencoder_1_16 --iter 200 --typeCom Workers

# Variational Autoencoder 2-16_8
python code/exp_level2_TFfederated_autoencoder.py statistics Variational_Autoencoder_2_16_8 --iter 200 --typeCom Workers

# KNN-50
python code/exp_level2_TFfederated_KNN.py statistics 50 --typeCom Workers --iter 200