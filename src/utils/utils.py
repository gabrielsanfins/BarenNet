import tensorflow as tf

def Create_Similarity_Model(n_nonsimilar, n_similar):
    '''
    
    Function that creates a neural network to discover incomplete similarities in the last n_similar paramaters of our data
    
    '''
    inputs = tf.keras.Input(shape = (n_nonsimilar +  n_similar,))

    similar_parameters = tf.keras.layers.Lambda(lambda x : x[:, n_nonsimilar:])(inputs)
    nonsimilar_parameters = [0 for i in range(n_nonsimilar)]
    Phi1_lists_nonsimilar = [0 for i in range(n_nonsimilar)]
    Phi1_lists_similar = [0 for i in range(n_nonsimilar)]
    Phi1_input_list = [0 for i in range(n_nonsimilar)]

    # build the exponential layer to represent the inputs of our Phi_1 function
    for i in range(n_nonsimilar):
        nonsimilar_parameters[i] = tf.keras.layers.Lambda(lambda x : x[:,i:i+1])(inputs)
        Phi1_lists_nonsimilar[i] = tf.keras.layers.Dense(1, activation = 'exponential', use_bias = False, name = 'similarity_layer_' + str(i+1))(similar_parameters)
        Phi1_lists_similar[i] = tf.keras.layers.Lambda(lambda x : tf.exp(x))(nonsimilar_parameters[i])
        Phi1_input_list[i] = tf.keras.layers.Multiply()([Phi1_lists_nonsimilar[i], Phi1_lists_similar[i]])
    Phi1_input = tf.keras.layers.Concatenate()(Phi1_input_list)

    # build the dense layers in order to fit our similarity function
    dense1 = tf.keras.layers.Dense(128, activation = 'relu')(Phi1_input)
    dense2 = tf.keras.layers.Dense(64, activation = 'relu')(dense1)
    dense3 = tf.keras.layers.Dense(32, activation = 'relu')(dense2)
    phi1 = tf.keras.layers.Dense(1, activation = 'relu')(dense3)

    # build the layer for the multiplication of the similar parameters
    multiplication_layer = tf.keras.layers.Dense(1, activation = 'exponential', use_bias = False, name = 'multiplication_layer')(similar_parameters)

    outputs = tf.keras.layers.Multiply()([phi1, multiplication_layer])

    model = tf.keras.Model(inputs = inputs, outputs = outputs)

    return model