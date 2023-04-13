import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from keras import Model
from keras.layers import Dense, Input, Concatenate, Reshape
from keras.layers.recurrent import LSTM
from keras.utils.vis_utils import plot_model


DATA_PATH = os.path.join("/dcs/large/u2288122/Workspace/location-tensors/data/")
TRACKING_DATA_PATH = os.path.join(DATA_PATH, "tracking_data")
CSV_DATA_PATH = os.path.join(TRACKING_DATA_PATH, "csv_data")
TENSOR_DATA_PATH = os.path.join(TRACKING_DATA_PATH, "tensor_data")
N_INPUT_TSTEPS = 4
N_OUTPUT_TSTEPS = 12
TRAIN_TEST_SPLIT = 0.25
RANDOM_SEED = 47
NUM_CAMS = 16

'''
CAVEAT: Using n + 1 cameras
'''


def load_dataset(dataset_path, split_fraction=TRAIN_TEST_SPLIT, seed=RANDOM_SEED):
    # Load, split and return the training and testing data
    X, Y = load_pkl(dataset_path)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_fraction, random_state=seed)
    Y_train_encoded = tensor_encode_one_hot(Y_train)
    Y_test_encoded = tensor_encode_one_hot(Y_test)
    return [tf.convert_to_tensor(array, dtype=tf.float32) for array in  [X_train, Y_train, Y_train_encoded, X_test, Y_test, Y_test_encoded]]


def tensor_encode_one_hot(tensor):
    one_hot_tensor = np.vstack([    \
        np.expand_dims( \
            np.hstack([ \
                tf.one_hot(example[:, 0], depth=NUM_CAMS), example[:, 1:]] \
            ), axis=0)
                for example in tensor])
    return one_hot_tensor


def tensor_decode_one_hot(one_hot_tensor):
    tensor = np.vstack([    \
        np.expand_dims( \
            np.hstack([ \
                np.expand_dims(np.argmax(example[:, 0:NUM_CAMS], axis=1), axis=1)   \
                , example[:, NUM_CAMS:]] \
            ), axis=0)
                for example in one_hot_tensor])
    return tensor


def store_pkl(object, output_file):
    try:
        with open(output_file, 'wb') as pkl_file:
            pickle.dump(object, pkl_file)
    except:
        print("Error storing data pickle.")
        raise


def load_pkl(input_file):
    try:
        with open(input_file, 'rb') as pkl_file:
            return pickle.load(pkl_file)
    except:
        print("Error storing data pickle.")
        raise

def combined_loss_fn(Y, Y_pred):
    CCE_loss = tf.keras.losses.CategoricalCrossentropy()
    MSE_loss = tf.keras.losses.MeanSquaredError()
    agg_loss = CCE_loss(Y[:, 0: NUM_CAMS + 4], Y_pred[:, 0: NUM_CAMS + 4])  \
                + MSE_loss(Y[:, NUM_CAMS + 4:], Y_pred[:, NUM_CAMS + 4:])
    return(agg_loss)


if __name__ == '__main__':

    # Load the dataset
    X_train, Y_train, Y_train_encoded, X_test, Y_test, Y_test_encoded = load_dataset(   \
        os.path.join(TENSOR_DATA_PATH, f"tensors_{N_INPUT_TSTEPS}_in_{N_OUTPUT_TSTEPS}_out_dataset.pkl"))
    
    print("Dataset loaded successfully.")
    print(f"\t\tX\t\t\tY\t\tY_encoded")
    print(f"train\t{X_train.shape}\t{Y_train.shape}\t{Y_train_encoded.shape}")
    print(f"train\t{X_test.shape}\t{Y_test.shape}\t{Y_test_encoded.shape}")
    
    # Define the model architecture
    #TO-DO: try adding a common dense layer; vary the lstm output shape; provide k as categorical
    intermediate_layer = Input(shape=(N_OUTPUT_TSTEPS))
    classifier = Dense(units=NUM_CAMS, activation='softmax')(intermediate_layer)
    regressor = Dense(units=4)(intermediate_layer)
    repeated_sub_unit = Model(intermediate_layer, [classifier, regressor])
    repeated_sub_unit.compile(optimizer='adam', 
                loss = {'classifier':'categorical_crossentropy', 'regressor':'mse'},
                loss_weights = {'classifier':1., 'regressor':0.5})
    # summarize layers
    print(repeated_sub_unit.summary())

    # Main network
    input_layer = Input(shape=(N_INPUT_TSTEPS, 5))
    intermediate_layer = LSTM(units=N_OUTPUT_TSTEPS, input_shape=(N_INPUT_TSTEPS, 5), \
                            activation="tanh", recurrent_activation="tanh")(input_layer)
    downstream_network = [Concatenate()(repeated_sub_unit(intermediate_layer)) for _ in range(N_OUTPUT_TSTEPS)]
    merge_layer = Reshape((N_OUTPUT_TSTEPS, NUM_CAMS + 4))(Concatenate()(downstream_network))
    model = Model(input_layer, merge_layer)
    model.compile(optimizer='adam', loss=combined_loss_fn, metrics=['accuracy'])

    # summarize layers
    print(model.summary())
    plot_model(model, to_file="model.png", show_shapes=True)
    
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

    #TO-Do: tensor2trajectory -> visualize