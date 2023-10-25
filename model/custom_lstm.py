#External imports
import numpy as np
import os
import pickle
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow import keras
from keras import Model
from keras.models import Sequential
from keras.layers import Dense, Input, Layer, LSTMCell, Reshape
from keras.utils.vis_utils import plot_model

#Internal imports
from global_config.global_config import (
    MODEL_PATH,
    N_INPUT_TSTEPS, N_OUTPUT_TSTEPS, NUM_CAMS, NUM_FEATURES)
from loader.loader import load_dataset
from utils.utils import (
    tensor_decode_one_hot, decode_2d_one_hot)
'''
CAVEAT: Using n + 1 cameras
'''

class prediction_block(Layer):
    def __init__(self, num_classes=NUM_CAMS):
        super(prediction_block, self).__init__()
        self.classifier = Dense(units=num_classes, activation='softmax')
        self.regression_activation = Dense(units=5, activation='tanh')
        self.regressor = Dense(units=4)

    def call(self, inputs):
        camera_classes = self.classifier(inputs)
        positions = self.regressor(self.regression_activation(inputs))
        return tf.concat([camera_classes, positions], axis=1)

class lstm_cell(Layer):
    def __init__(self, lstm_units=NUM_FEATURES,num_cams=NUM_CAMS):
        super(lstm_cell, self).__init__()
        self.base_cell = LSTMCell(units=lstm_units)
        self.predictor = prediction_block(num_classes=num_cams)
        self.state_size = lstm_units
        self.encoded_output_size = lstm_units + num_cams - 1

    """def get_initial_state(self, inputs):
        batch_size = inputs.shape[0]
        dtype = inputs.dtype
        h_prev = tf.zeros((batch_size, self.state_size), dtype=dtype)
        c_prev = tf.zeros((batch_size, self.state_size), dtype=dtype)
        return [h_prev, c_prev]

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        LSTMCell.get_initial_state(inputs=inputs, batch_size=batch_size, dtype=dtype)

        """

    def call(self, inputs, states):
        if states is None:
            # Initialize the first layers state to zeros.
            # h_prev, c_prev = self.get_initial_state(inputs)
            h_prev, c_prev = self.base_cell.get_initial_state(inputs)
        else:
            h_prev, c_prev = states
        
        h, new_states = self.base_cell(inputs, [h_prev, c_prev])
        y_encoded = self.predictor(h)
        
        _, c = new_states # Discard intermediate h
        h = decode_2d_one_hot(y_encoded)
        return y_encoded, [h, c]
    

def combined_loss_fn(Y, Y_pred, num_cams=NUM_CAMS):
    cam_loss = tf.keras.losses.CategoricalCrossentropy()
    box_loss = tfa.losses.GIoULoss()
    agg_loss = cam_loss(Y[:, :, 0: num_cams], Y_pred[:, :, 0: num_cams])  \
                + 0.001 * box_loss(Y[:, :, num_cams:], Y_pred[:, :, num_cams:])
    return(agg_loss)


def save_model(model, name="lstm_giou.ml", path=MODEL_PATH):
    model.save(os.path.join(MODEL_PATH, name))


def load_model(name, path=MODEL_PATH):
    return tf.keras.models.load_model(os.path.join(MODEL_PATH, name), custom_objects={'combined_loss_fn': combined_loss_fn})


def define_model(n_input_tsteps=N_INPUT_TSTEPS, n_output_tsteps=N_OUTPUT_TSTEPS, num_cams=NUM_CAMS):
    """Define the model architecture"""
    #TO-DO: try adding a common dense layer; vary the lstm output shape;
    # Main network
    input_layer = Input(shape=(n_input_tsteps, 5))
    
    y = None
    Y = []
    state = None

    for t_step in range(n_input_tsteps):
        y, state = lstm_cell()(input_layer[:, t_step, :], state)
        
    Y.append(y)

    for _ in range(n_output_tsteps - 1):
        x, _ = state
        y, state = lstm_cell()(x, state)
        Y.append(y)


    merge_layer = Reshape((n_output_tsteps, num_cams + 4))(
        tf.concat(Y, axis=1))
    # merge_layer = Reshape((len(Y), num_cams + 4))(tf.concat(Y, axis=0))
    model = Model(input_layer, merge_layer)
    model.compile(optimizer='adam', loss=combined_loss_fn, metrics=['cosine_similarity'])

    # summarize layers
    print(f"Model summary:\n{model.summary()}")
    plot_model(model, to_file="model.png", show_shapes=True)
    return model


def train_model(model=None, dataset=None):
    # TO-DO: Globalise batch_size, epochs
    # Load the dataset
    if dataset is None:
        X_train, Y_train, Y_train_encoded, X_test, Y_test, Y_test_encoded = load_dataset()
    else:
        X_train, Y_train, Y_train_encoded, X_test, Y_test, Y_test_encoded = dataset

    print(f"\t\tX\t\t\tY\t\tY_encoded")
    print(f"train\t{X_train.shape}\t{Y_train.shape}\t{Y_train_encoded.shape}")
    print(f"train\t{X_test.shape}\t{Y_test.shape}\t{Y_test_encoded.shape}")
    
    # Load the model
    if model is None:
        model = define_model()

    # Training 
    print("Training the model...")
    logs = model.fit(X_train, Y_train_encoded, batch_size=16, epochs=50)
    print("Model training completed.")
    print(logs.history)

    # Evaluation
    print("Testing the model...")
    results = model.evaluate(X_test, Y_test_encoded, batch_size=128)
    print("Results")
    print(results)
    model.predict(X_test)
    return model, results, logs


def predict(model, X_test, Y_test_encoded, debug=False):
    predictions = model.predict(X_test)
    y_pred = tensor_decode_one_hot(predictions)
    if debug:
        # print(predictions[:, :, 0:NUM_CAMS])
        print(y_pred[:, :, 0])
        print(tensor_decode_one_hot(Y_test_encoded)[:, :, 0])
        return y_pred.numpy().astype(int), predictions
    return y_pred.numpy().astype(int)
