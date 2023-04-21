#External imports
import numpy as np
import os
import pickle
import tensorflow as tf

from tensorflow import keras
from keras import Model
from keras.layers import Dense, Input, Concatenate, Reshape
from keras.layers.recurrent import LSTM
from keras.utils.vis_utils import plot_model

#Internal imports
from global_config.global_config import (
    MODEL_PATH,
    N_INPUT_TSTEPS, N_OUTPUT_TSTEPS, NUM_CAMS)
from loader.loader import load_dataset
from utils.utils import (
    tensor_encode_one_hot, tensor_decode_one_hot)
'''
CAVEAT: Using n + 1 cameras
'''

def combined_loss_fn(Y, Y_pred):
    CCE_loss = tf.keras.losses.CategoricalCrossentropy()
    MSE_loss = tf.keras.losses.MeanSquaredError()
    agg_loss = 10 * CCE_loss(Y[:, 0: NUM_CAMS + 4], Y_pred[:, 0: NUM_CAMS + 4])  \
                + MSE_loss(Y[:, NUM_CAMS + 4:], Y_pred[:, NUM_CAMS + 4:])
    return(agg_loss)


def save_model(model, name, path=MODEL_PATH):
    model.save(os.path.join(MODEL_PATH, name))


def load_model(name, path=MODEL_PATH):
    return tf.keras.models.load_model(os.path.join(MODEL_PATH, name), custom_objects={'combined_loss_fn': combined_loss_fn})


def define_model(n_input_tsteps=N_INPUT_TSTEPS, n_output_tsteps=N_OUTPUT_TSTEPS, num_cams=NUM_CAMS):
    """Define the model architecture"""
    #TO-DO: try adding a common dense layer; vary the lstm output shape;
    intermediate_layer = Input(shape=(n_output_tsteps))
    classifier = Dense(units=num_cams, activation='softmax')(intermediate_layer)
    regressor = Dense(units=4)(intermediate_layer)
    repeated_sub_unit = Model(intermediate_layer, [classifier, regressor])
    repeated_sub_unit.compile(optimizer='adam', 
                loss = {'classifier':'categorical_crossentropy', 'regressor':'mse'},
                loss_weights = {'classifier':1., 'regressor':0.5})
    
    # Main network
    input_layer = Input(shape=(n_input_tsteps, 5))
    intermediate_layer = LSTM(units=n_output_tsteps, input_shape=(n_input_tsteps, 5), \
                            activation="tanh", recurrent_activation="tanh")(input_layer)
    downstream_network = [Concatenate()(repeated_sub_unit(intermediate_layer)) for _ in range(n_output_tsteps)]
    merge_layer = Reshape((n_output_tsteps, num_cams + 4))(Concatenate()(downstream_network))
    model = Model(input_layer, merge_layer)
    model.compile(optimizer='adam', loss=combined_loss_fn, metrics=['cosine_similarity'])

    # summarize layers
    print(f"Model summary:\n{model.summary()}")
    print(f"Sub_unit summary:\n{repeated_sub_unit.summary()}")
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
    logs = model.fit(X_train, Y_train_encoded, batch_size=32, epochs=10)
    print("Model training completed.")
    print(logs.history)

    # Evaluation
    print("Testing the model...")
    results = model.evaluate(X_test, Y_test_encoded, batch_size=128)
    print("Results")
    print(results)
    predictions = model.predict(X_test)
    y_pred = tensor_decode_one_hot(predictions.astype(int))
    print(y_pred[:, :, 0])
    return model, results, logs


def predict(model, X_test, Y_test_encoded):
    predictions = model.predict(X_test)
    return predictions
    #To-Do: tensor2trajectory -> visualize