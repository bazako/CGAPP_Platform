# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

def autoencoder_skeleton_1D(inputDim,lantenDim): 
    return tf.keras.Sequential([
        tf.keras.layers.Dense(lantenDim, activation='relu',input_shape=(inputDim,), kernel_initializer='glorot_uniform', bias_initializer='zeros'),
        tf.keras.layers.Dense(inputDim, activation='sigmoid',kernel_initializer='glorot_uniform', bias_initializer='zeros')
        ])
    
def autoencoder_skeleton_2D(inputDim,hidden,latentDim): 
    return tf.keras.Sequential([
        tf.keras.layers.Dense(hidden, activation='relu',input_shape=(inputDim,),kernel_initializer='glorot_uniform', bias_initializer='zeros'),
        tf.keras.layers.Dense(latentDim, activation='relu',kernel_initializer='glorot_uniform', bias_initializer='zeros'),
        tf.keras.layers.Dense(hidden, activation='relu',kernel_initializer='glorot_uniform', bias_initializer='zeros'),
        tf.keras.layers.Dense(inputDim, activation='sigmoid',kernel_initializer='glorot_uniform', bias_initializer='zeros')
        ])


def sampling(mu_log_variance):
    mu, log_variance = mu_log_variance
    epsilon = tf.keras.backend.random_normal(shape=tf.keras.backend.shape(mu), mean=0.0, stddev=1.0)
    random_sample = mu + tf.keras.backend.exp(log_variance/2) * epsilon
    return random_sample

def loss_func(encoder_mu, encoder_log_variance):
    def vae_reconstruction_loss(y_true, y_predict):
        reconstruction_loss_factor = 1000
        reconstruction_loss = tf.keras.backend.mean(tf.keras.backend.square(y_true-y_predict), axis=[1])
        return reconstruction_loss_factor * reconstruction_loss

    def vae_kl_loss(encoder_mu, encoder_log_variance):
        kl_loss = -0.5 * tf.keras.backend.sum(1.0 + encoder_log_variance - tf.keras.backend.square(encoder_mu) - tf.keras.backend.exp(encoder_log_variance), axis=1)
        return kl_loss

    def vae_kl_loss_metric(y_true, y_predict):
        kl_loss = -0.5 * tf.keras.backend.sum(1.0 + encoder_log_variance - tf.keras.backend.square(encoder_mu) - tf.keras.backend.exp(encoder_log_variance), axis=1)
        return kl_loss

    def vae_loss(y_true, y_predict):
        reconstruction_loss = vae_reconstruction_loss(y_true, y_predict)
        kl_loss = vae_kl_loss(y_true, y_predict)

        loss = reconstruction_loss + kl_loss
        return loss

    return vae_loss


    
def variational_autoencoder_skeleton_2D(inputDim,hidden,latent_dim): 
    
    encoder_inputs = keras.Input(shape=(inputDim,))
    x1 = tf.keras.layers.Dense(hidden, activation='relu',kernel_initializer='glorot_uniform', bias_initializer='zeros', name='hidden')(encoder_inputs)
    x2 = tf.keras.layers.Dense(latent_dim, activation='relu',kernel_initializer='glorot_uniform', bias_initializer='zeros', name='latent')(x1)  
    
    encoder_mu = layers.Dense(latent_dim, name="z_mean")(x2)
    encoder_log_variance = layers.Dense(latent_dim, name="z_log_var")(x2)
    
    encoder_output = tf.keras.layers.Lambda(sampling, name="encoder_output")([encoder_mu, encoder_log_variance])
    encoder = tf.keras.models.Model(encoder_inputs, encoder_output, name="encoder_model")
                                 
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(hidden, activation='relu',kernel_initializer='glorot_uniform', bias_initializer='zeros')(latent_inputs)
    decoder_outputs = tf.keras.layers.Dense(inputDim, activation='sigmoid',kernel_initializer='glorot_uniform', bias_initializer='zeros')(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    
    
    vae_input = keras.Input(shape=(inputDim,))
    vae_encoder_output = encoder(vae_input)
    vae_decoder_output = decoder(vae_encoder_output)
    vae = tf.keras.models.Model(vae_input, vae_decoder_output, name="VAE")
    
    

    
    
    
    return vae, loss_func(encoder_mu, encoder_log_variance)

def variational_autoencoder_skeleton_1D(inputDim,latent_dim): 
    
    encoder_inputs = keras.Input(shape=(inputDim,))
    x = tf.keras.layers.Dense(latent_dim, activation='relu',kernel_initializer='glorot_uniform', bias_initializer='zeros', name='latent')(encoder_inputs)
    
    encoder_mu = layers.Dense(latent_dim, name="z_mean")(x)
    encoder_log_variance = layers.Dense(latent_dim, name="z_log_var")(x)
    
    encoder_output = tf.keras.layers.Lambda(sampling, name="encoder_output")([encoder_mu, encoder_log_variance])
    encoder = tf.keras.models.Model(encoder_inputs, encoder_output, name="encoder_model")
                                 
    latent_inputs = keras.Input(shape=(latent_dim,))
    decoder_outputs = tf.keras.layers.Dense(inputDim, activation='sigmoid',kernel_initializer='glorot_uniform', bias_initializer='zeros')(latent_inputs)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    
    
    vae_input = keras.Input(shape=(inputDim,))
    vae_encoder_output = encoder(vae_input)
    vae_decoder_output = decoder(vae_encoder_output)
    vae = tf.keras.models.Model(vae_input, vae_decoder_output, name="VAE")
    
    

    
    
    
    return vae, loss_func(encoder_mu, encoder_log_variance)
  