# Code and algorithm design by Paolo Fazzini (paolo.fazzini@cnr.it)

import os

# Get the name of the current Conda environment
conda_env = os.environ.get('CONDA_DEFAULT_ENV')




import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, SimpleRNN, LSTM, GRU, Flatten




class ANNModel:
    def __init__(
                    self, name, type_cell, units, in_vars, out_vars, 
                    out_steps, activation, epochs, labels, 
                    heuristic_weights, heuristic_method, weight_init=0, ensemble=True
                ):
        self.name = name
        self.hyperparameters = {
            'type_cell': type_cell,
            'units': units,
            'in_vars': in_vars,
            'out_vars': out_vars,
            'out_steps': out_steps,
            'activation': activation,
            'epochs': epochs
        }
        self.ANNModel_obj = None
        self.loss = None
        self.predict_direct = []
        self.predict_recursive = []
        self.loss_history = None
        self.labels = labels
        self.ensemble = ensemble
        self.heuristic_weights = heuristic_weights
        self.heuristic_method = heuristic_method
        self.weight_init = weight_init
        self.units = units




def custom_loss(y_true, y_pred):
    mse_loss_func = tf.keras.losses.MeanSquaredError() 
    mse_loss = mse_loss_func(y_true, y_pred)

    entropy_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))  # Assuming binary entropy
    alpha = 1  # Weight for MSE loss
    beta = 1 - alpha  # Weight for entropy loss
    combined_loss = alpha * mse_loss + beta * entropy_loss

    return combined_loss




def _set_cell(type_ann, units, input_shape, activation):
    if type_ann == "rnn":
        return SimpleRNN(units, input_shape=input_shape, activation=activation)
    elif type_ann == "gru":
        return GRU(units, input_shape=input_shape, activation=activation)
    elif type_ann == "lstm":
        return LSTM(units, input_shape=input_shape, activation=activation)
    elif type_ann == "mlp":
        return Dense(units, input_shape=input_shape, activation=activation)

def create_model(ANNModel_obj, input_shape):
    # NOTE: for the time being this only works with an LSTM based ANN, which has
    # 4 gates (hence 4 is hardcoded)

    # Input layer
    i_layer = Input(input_shape)
    
    # RNN/ANN layer
    m_layer = _set_cell(ANNModel_obj.hyperparameters['type_cell'], ANNModel_obj.hyperparameters['units'], input_shape, ANNModel_obj.hyperparameters['activation'][0])(i_layer)
    
    if ANNModel_obj.hyperparameters['type_cell'] == 'mlp':
        m_layer = Flatten()(m_layer)

    # Create the output layers first (to ensure model is created)
    o_layers = []
    for _ in range(ANNModel_obj.hyperparameters['out_vars']):
        o_layers.append(Dense(units=ANNModel_obj.hyperparameters['out_steps'], activation=ANNModel_obj.hyperparameters['activation'][1])(m_layer))
    
    # Create the model
    ANNModel_obj.ANN_model = Model(inputs=i_layer, outputs=o_layers)
    ANNModel_obj.loss = custom_loss
    
    


