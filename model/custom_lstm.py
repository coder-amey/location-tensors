#External imports
import numpy as np
import os
import pickle
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow import keras
from keras import Model
from keras.layers import Dense, Input, Layer, LSTM, LSTMCell, Reshape, RNN
from keras.utils.vis_utils import plot_model

#Internal imports
from global_config.global_config import (
    MODEL_PATH,
    N_INPUT_TSTEPS, N_OUTPUT_TSTEPS, NUM_CAMS, NUM_FEATURES,
    EPOCHS, TRAIN_BATCH_SIZE, TEST_BATCH_SIZE)
from loader.loader import load_dataset
from utils.utils import (
    tensor_decode_one_hot, decode_2d_one_hot)
'''
CAVEAT: Using n + 1 cameras
'''

class prediction_block(Layer):
    def __init__(self, num_classes=NUM_CAMS, num_features=NUM_FEATURES):
        super(prediction_block, self).__init__()
        self.classifier = Dense(units=num_classes, activation='softmax')
        self.regression_activation = Dense(units=num_features, activation='sigmoid')
        self.regressor = Dense(units=4)

    def call(self, inputs):
        camera_classes = self.classifier(inputs)
        positions = self.regressor(self.regression_activation(inputs))
        return tf.concat([camera_classes, positions], axis=1)

class lstm_cell(LSTMCell):
    def __init__(self, units=NUM_FEATURES, num_cams=NUM_CAMS, **kwargs):
        self.state_size = units
        self.output_size = units + num_cams - 1
        self.base_cell = LSTMCell(units=units)
        self.predictor = prediction_block(num_classes=num_cams, num_features=units)
        super(lstm_cell, self).__init__(units=units, **kwargs)

    def call(self, inputs, states=None):
        if states is None:
            # Initialize the first layers state to zeros.
            states = super(lstm_cell, self).get_initial_state(inputs)
        
        h, states = self.base_cell(inputs, states)
        y_encoded = self.predictor(h)
        
        return y_encoded, states


class decoder_rnn(Layer):
    '''Returns sequence of predicted location_tensors of shape:
        (batch_size, n_output_tsteps, (num_cams + num_features))'''
    def __init__(self, n_output_steps=N_OUTPUT_TSTEPS, num_features=NUM_FEATURES, num_cams=NUM_CAMS):
        super(decoder_rnn, self).__init__()
        self.decoder_units = n_output_steps
        self.units = num_features
        self.num_cams=num_cams
        self.rnn = [lstm_cell(units=self.units,num_cams=self.num_cams) \
            for _ in range(self.decoder_units)]
    
   
    def call(self, x, state):
        # Output -> Input latch loop
        Y = []
        for cell in self.rnn:
            y, h, c = RNN(cell, return_state=True)(x, initial_state=state)
            Y.append(y)
            state = [h, c]
            x = tf.expand_dims(decode_2d_one_hot(y), axis=1)
        # Concatenate and return the sequence output
        return tf.stack(Y, axis=1)
        
def combined_loss_fn(Y, Y_pred, num_cams=NUM_CAMS):
    cam_loss = tf.keras.losses.CategoricalCrossentropy()    # ~ 2.0
    box_loss = tfa.losses.GIoULoss()                        # < 2.0
    agg_loss = 4 * cam_loss(Y[:, :, 0: num_cams], Y_pred[:, :, 0: num_cams]) + \
        10 * box_loss(Y[:, :, num_cams:], Y_pred[:, :, num_cams:])
    return(agg_loss)


def save_model(model, name="lstm_giou.ml", path=MODEL_PATH):
    model.save(os.path.join(MODEL_PATH, name))


def load_model(name, path=MODEL_PATH):
    return tf.keras.models.load_model(os.path.join(MODEL_PATH, name), custom_objects={'combined_loss_fn': combined_loss_fn})


def define_model(n_input_tsteps=N_INPUT_TSTEPS, n_output_tsteps=N_OUTPUT_TSTEPS, num_features=NUM_FEATURES, num_cams=NUM_CAMS):
    """Define the model architecture"""
    #TO-DO: try adding a common dense layer; vary the lstm output shape;
    encoder_lstm = LSTM(units=num_features, return_state=True)
    decoder_lstm = decoder_rnn(n_output_steps=n_output_tsteps, num_features=num_features, num_cams=num_cams)

    # Main network
    input_layer = Input(shape=(n_input_tsteps, num_features))
    # Encoder LSTM
    encoder_layer, h, c = encoder_lstm(input_layer)
    encoder_layer = Reshape((1, num_features))(encoder_layer) # [!] VERY IMPORTANT TO INCLUDE A TIME-STEP
    print(input_layer.shape)
    print(encoder_layer.shape)
    # Decoder LSTM
    decoder_layer = decoder_lstm(encoder_layer, [h, c])
    
    # Compile model
    model = Model(input_layer, decoder_layer)
    model.compile(optimizer='adam', loss=combined_loss_fn, metrics=['cosine_similarity'])

    # summarize layers
    print(f"Model summary:\n{model.summary()}")
    plot_model(model, to_file="model.png", show_shapes=True)
    return model


def train_model(model=None, dataset=None, epochs=EPOCHS, train_batch_size=TRAIN_BATCH_SIZE, test_batch_size=TEST_BATCH_SIZE, num_cams=NUM_CAMS):
    # Load the dataset
    if dataset is None:
        X_train, Y_train, Y_train_encoded, X_test, Y_test, Y_test_encoded = load_dataset()
    else:
        X_train, Y_train, Y_train_encoded, X_test, Y_test, Y_test_encoded = dataset

    print(f"\t\tX\t\t\tY\t\tY_encoded")
    print(f"train\t{X_train.shape}\t{Y_train.shape}\t{Y_train_encoded.shape}")
    print(f"test\t{X_test.shape}\t{Y_test.shape}\t{Y_test_encoded.shape}")
    
    # Load the model
    if model is None:
        model = define_model()

    # Training 
    print("Training the model...")
    logs = model.fit(X_train, Y_train_encoded, batch_size=train_batch_size, epochs=epochs)
    print("Model training completed.")
    # print(logs.history)

    # Evaluation
    print("Testing the model...")
    results = model.evaluate(X_test, Y_test_encoded, batch_size=test_batch_size)
    # logs["test_loss"] = results

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
