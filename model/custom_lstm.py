#External imports
import numpy as np
import os
import pickle
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow import keras
from keras import Model
from keras.layers import Dense, Input, Layer, LSTM, LSTMCell, Reshape, RNN
from tensorflow.keras.metrics import Precision
from keras.utils.vis_utils import plot_model

#Internal imports
from global_config.global_config import (
	MODEL_PATH,
	N_INPUT_TSTEPS, N_OUTPUT_TSTEPS, NUM_CAMS, NUM_FEATURES,
	EPOCHS, TRAIN_BATCH_SIZE, TEST_BATCH_SIZE,
	CAM_LOSS, BOX_LOSS)

from loader.loader import load_dataset
from utils.utils import (
	tensor_decode_one_hot, generate_targets, targets2tensors)
'''
CAVEAT: Using n + 1 cameras
'''


# Retained for compatibility
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
	predictions = []

	# Main network
	print("Building model...")
	input_layer = Input(shape=(n_input_tsteps, num_features))
	# Encoder LSTM
	encoder_lstm = LSTM(units=num_features, return_state=True)
	encoder_layer, h, c = encoder_lstm(input_layer)
	x = Reshape((1, num_features))(encoder_layer) # [!] VERY IMPORTANT TO INCLUDE A TIME-STEP
	# Decoder LSTM
	for block_index in range(n_output_tsteps):
		y, h, c = RNN(LSTMCell(units=num_features), \
					  return_state=True)(x, initial_state=[h, c])

		# Prediction module
		intermediate_layer = Dense(units=num_features, activation='sigmoid')(y)
		classifier = Dense(units=num_cams, activation='softmax', name=f"classifier_{block_index}")
		cam_pred = classifier(intermediate_layer)
		predictions.append(cam_pred)

		regressor = Dense(units=4, activation='linear', name=f"regressor_{block_index}")
		box_pred = regressor(intermediate_layer)
		predictions.append(box_pred)

		x = Reshape((1, num_features))(y) # [!] VERY IMPORTANT TO INCLUDE A TIME-STEP

	# Compile model
	loss = {}
	for i in range(n_output_tsteps):
		loss[f"classifier_{i}"] = CAM_LOSS
		loss[f"regressor_{i}"] = BOX_LOSS

	model = Model(inputs=input_layer, outputs=predictions)
	model.compile(optimizer='adam', loss=loss) #, metrics=[Precision()])

	# Summarize layers
	print(f"Model compiled successfully.\nModel summary:")
	model.summary()
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

	targets_train = generate_targets(Y_train_encoded)
	targets_test = generate_targets(Y_test_encoded)
	print("Targets loaded successfully")

	# Load the model
	if model is None:
		model = define_model()

	# Training
	print("Training the model...")
	logs = model.fit(X_train, targets_train, batch_size=train_batch_size, epochs=epochs, verbose=2)
	print("Model training completed.")
	# print(logs.history)

	# Evaluation
	results = model.evaluate(X_test, targets_test, batch_size=test_batch_size)
	# logs["test_loss"] = results

	return model, results, logs


def predict(model, X_test, Y_test_encoded, debug=False):
	predictions = model.predict(X_test)
	y_pred = tensor_decode_one_hot(targets2tensors(predictions))
	if debug:
		# print(predictions[:, :, 0:NUM_CAMS])
		print(y_pred[:, :, 0])
		print(tensor_decode_one_hot(Y_test_encoded)[:, :, 0])
		return y_pred.numpy().astype(int), predictions
	return y_pred.numpy().astype(int)
