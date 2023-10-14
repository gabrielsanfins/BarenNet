import tensorflow as tf
import pandas as pd
import numpy as np
from typing import List, Tuple
# from tensorflow.keras import regularizers


def Create_Similarity_Model(n_nonsimilar, n_similar):
    '''

    Function that creates a neural network to discover incomplete similarities in the last n_similar paramaters of a dataset

    '''

    inputs = tf.keras.Input(shape=(n_nonsimilar + n_similar,))

    similar_parameters = tf.keras.layers.Lambda(lambda x : x[:, n_nonsimilar:])(inputs)
    nonsimilar_parameters = tf.keras.layers.Lambda(lambda x : x[:, :n_nonsimilar])(inputs)
    Phi1_lists_nonsimilar = tf.keras.layers.Dense(n_nonsimilar, activation = 'exponential', use_bias = False, name = 'similarity_layer')(similar_parameters)
    Phi1_lists_similar = tf.keras.layers.Lambda(lambda x : tf.exp(x))(nonsimilar_parameters)
    Phi1_input = tf.keras.layers.Multiply()([Phi1_lists_nonsimilar, Phi1_lists_similar])

    # Without batch normalization

    activation_function = 'relu'

    dense1 = tf.keras.layers.Dense(256, activation = activation_function)(Phi1_input)
    dense2 = tf.keras.layers.Dense(128, activation = activation_function)(dense1)
    dense3 = tf.keras.layers.Dense(64, activation = activation_function)(dense2)
    dense4 = tf.keras.layers.Dense(32, activation = activation_function)(dense3)
    dense5 = tf.keras.layers.Dense(16, activation = activation_function)(dense4)
    phi1 = tf.keras.layers.Dense(1, activation = None)(dense5)


    # build the layer for the multiplication of the similar parameters
    multiplication_layer = tf.keras.layers.Dense(1, activation = None, use_bias = False, name = 'multiplication_layer')(similar_parameters)

    outputs = tf.keras.layers.Add()([phi1, multiplication_layer])

    model = tf.keras.Model(inputs = inputs, outputs = outputs)

    return model

    # # build the dense layers in order to fit our similarity function
    # dense1 = tf.keras.layers.Dense(256, activation = 'gelu')(Phi1_input)
    # bn1 = tf.keras.layers.BatchNormalization()(dense1)
    # dense2 = tf.keras.layers.Dense(128, activation = 'gelu')(bn1)
    # bn2 = tf.keras.layers.BatchNormalization()(dense2)
    # dense3 = tf.keras.layers.Dense(64, activation = 'gelu')(bn2)
    # bn3 = tf.keras.layers.BatchNormalization()(dense3)
    # dense4 = tf.keras.layers.Dense(32, activation = 'gelu')(bn3)
    # bn4 = tf.keras.layers.BatchNormalization()(dense4)
    # dense5 = tf.keras.layers.Dense(16, activation = 'gelu')(bn4)
    # bn5 = tf.keras.layers.BatchNormalization()(dense5)
    # phi1 = tf.keras.layers.Dense(1, activation = None)(bn5)


def adjust_dataframe_according_to_similarity(
        data_path: str,
        non_similar_params: List[str],
        similar_params: List[str],
        non_dimensional_qoi: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """

    Makes the appropriate ordering of the Dataframe collumns so that it follows
    the order specified in the non-similar and similar parameters lists.

    The method also makes the log transformation in the DataFrame in order to
    make it suitable for the BarenNet.

    """
    original_df = pd.read_excel(data_path)

    dic_x = {}
    dic_y = {}

    for label in non_similar_params:
        label_values = original_df.loc[:, label].to_numpy()
        log_values = np.log(label_values)
        dic_x[label] = log_values

    for label in similar_params:
        label_values = original_df.loc[:, label].to_numpy()
        log_values = np.log(label_values)
        dic_x[label] = log_values

    nd_qoi_label_values = original_df.loc[:, non_dimensional_qoi].to_numpy()
    nd_qoi_log_values = np.log(nd_qoi_label_values)
    dic_y[non_dimensional_qoi] = nd_qoi_log_values

    df_log_x = pd.DataFrame.from_dict(dic_x)
    df_log_y = pd.DataFrame.from_dict(dic_y)

    return (df_log_x, df_log_y)
