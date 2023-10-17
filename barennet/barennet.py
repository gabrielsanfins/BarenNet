import tensorflow as tf
# from tensorflow.keras import regularizers


def create_barennet(n_nonsimilar: int,
                    n_similar: int,
                    dense_activation: str = "relu"):
    '''

    Function that creates a neural network to discover incomplete similarities
    in the last n_similar paramaters of a dataset.

    '''

    inputs = tf.keras.Input(shape=(n_nonsimilar + n_similar,))

    similar_parameters = tf.keras.layers.Lambda(
        lambda x: x[:, n_nonsimilar:])(inputs)

    nonsimilar_parameters = tf.keras.layers.Lambda(
        lambda x: x[:, :n_nonsimilar])(inputs)

    Phi1_lists_nonsimilar = tf.keras.layers.Dense(
        n_nonsimilar, activation='exponential',
        use_bias=False,
        name='similarity_layer')(similar_parameters)

    Phi1_lists_similar = tf.keras.layers.Lambda(
        lambda x: tf.exp(x))(nonsimilar_parameters)

    Phi1_input = tf.keras.layers.Multiply()(
        [Phi1_lists_nonsimilar, Phi1_lists_similar])

    # Without batch normalization

    dense1 = tf.keras.layers.Dense(
        256,
        activation=dense_activation)(Phi1_input)

    dense2 = tf.keras.layers.Dense(
        128,
        activation=dense_activation)(dense1)

    dense3 = tf.keras.layers.Dense(
        64,
        activation=dense_activation)(dense2)

    dense4 = tf.keras.layers.Dense(
        32,
        activation=dense_activation)(dense3)

    dense5 = tf.keras.layers.Dense(
        16,
        activation=dense_activation)(dense4)

    phi1 = tf.keras.layers.Dense(1, activation=None)(dense5)

    # build the layer for the multiplication of the similar parameters
    multiplication_layer = tf.keras.layers.Dense(
        1,
        activation=None,
        use_bias=False,
        name='multiplication_layer')(similar_parameters)

    outputs = tf.keras.layers.Add()([phi1, multiplication_layer])

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

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
