import tensorflow as tf

def create_f_model(f_hidden_layer, f_hidden_units, x_dim, y_dim):
    f_model = tf.keras.Sequential()
    f_model.add(tf.keras.layers.InputLayer(input_shape=(x_dim,)),)
    for _ in range(f_hidden_layer):
        f_model.add(tf.keras.layers.Dense(f_hidden_units, activation='relu'))
    f_model.add(tf.keras.layers.Dense(y_dim, activation='sigmoid'))
    return f_model

def create_z_model(f_model, z_layer):
    z_model = tf.keras.Model(inputs=f_model.inputs, outputs=f_model.layers[z_layer].output)
    return z_model

def create_v_model(v_hidden_layer, v_hidden_units, z_dim, u_dim):
    v_model = tf.keras.Sequential()
    v_model.add(tf.keras.layers.InputLayer(input_shape=(z_dim,)),)
    for _ in range(v_hidden_layer):
        v_model.add(tf.keras.layers.Dense(v_hidden_units, activation='leaky_relu'))
    if u_dim == 1:
        v_model.add(tf.keras.layers.Dense(u_dim, activation='sigmoid'))
    else:
        v_model.add(tf.keras.layers.Dense(u_dim, activation='softmax'))
    return v_model