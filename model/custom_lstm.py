#External imports
import numpy as np
import os
import pickle
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow import keras
from keras import Model
from keras.layers import Dense, Input, Layer, LSTM, LSTMCell, Reshape, RNN
from keras.metrics import Precision
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model

#Internal imports
from global_config.global_config import (
	MODEL_PATH,
	N_INPUT_TSTEPS, N_OUTPUT_TSTEPS, NUM_CAMS, NUM_FEATURES,
	EPOCHS, TRAIN_BATCH_SIZE, TEST_BATCH_SIZE,
	CAM_LOSS, BOX_LOSS, CAM_LOSS_WT, BOX_LOSS_WT)

from loader.loader import load_dataset
from utils.utils import (
	store_pkl, load_pkl,
	tensor_encode_one_hot, tensor_decode_one_hot,
	generate_targets, targets2tensors)
'''
CAVEAT: Using n + 1 cameras
'''


# Retained for compatibility
def combined_loss_fn(Y, Y_pred, num_cams=NUM_CAMS):
	cam_loss = tf.keras.losses.CategoricalCrossentropy()	# ~ 2.0
	box_loss = tfa.losses.GIoULoss()						# < 2.0
	agg_loss = 4 * cam_loss(Y[:, :, 0: num_cams], Y_pred[:, :, 0: num_cams]) + \
		10 * box_loss(Y[:, :, num_cams:], Y_pred[:, :, num_cams:])
	return(agg_loss)


def save_model(model, logs=None, name="custom_lstm.ml", path=MODEL_PATH):
	model.save(os.path.join(path, name))
	if logs is not None:
		store_pkl(logs, os.path.join(path, name, "logs.pkl"))


def load_model(name, path=MODEL_PATH):
	model = tf.keras.models.load_model(os.path.join(path, name)) #, custom_objects={'combined_loss_fn': combined_loss_fn}) [for compatibility]
	try:
		logs = load_pkl(os.path.join(path, name, "logs.pkl"))
		return model, logs
	except:
		print(f"Could not find model logs.")
		return model, None


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

		regressor = Dense(units=4, activation='linear', kernel_regularizer=l2(0.01), name=f"regressor_{block_index}")
		box_pred = regressor(intermediate_layer)
		predictions.append(box_pred)

		x = Reshape((1, num_features))(y) # [!] VERY IMPORTANT TO INCLUDE A TIME-STEP

	# Compile model
	loss = {}
	loss_weights = {}
	for i in range(n_output_tsteps):
		loss[f"classifier_{i}"] = CAM_LOSS
		loss_weights[f"classifier_{i}"] = CAM_LOSS_WT
		loss[f"regressor_{i}"] = BOX_LOSS
		loss_weights[f"regressor_{i}"] = BOX_LOSS_WT

	model = Model(inputs=input_layer, outputs=predictions)
	model.compile(optimizer='adam', loss=loss, loss_weights=loss_weights) #, metrics=[Precision()])

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
	logs = {'train_log': logs.history, 'parameters': logs.params}
	print("Model training completed.")

	# Evaluation
	logs['test_log'] = {key: value for key, value in zip(logs['train_log'].keys(), \
		test_model(model, X_test, targets=targets_test, test_batch_size=test_batch_size))}

	return model, logs


def test_model(model, X_test, Y_test=None, Y_test_encoded=None, targets=None, test_batch_size=TEST_BATCH_SIZE):
	assert 1 == sum([1 for param in [Y_test, Y_test_encoded, targets] \
			if param is not None]), "Pass any one of the three arguments: \
				Y_test, Y_test_encoded or targets"
	print("Testing the model...")
	if targets is None:
		if Y_test_encoded is None:
			targets = generate_targets(tensor_encode_one_hot(Y_test))
		else:
			targets = generate_targets(Y_test_encoded)
	results = model.evaluate(X_test, targets, batch_size=test_batch_size)
	print(f"Results:\n{results}")
	return results

def predict(model, X):
	predictions = model.predict(X)
	y_pred = tensor_decode_one_hot(targets2tensors(predictions))
	return y_pred.numpy().astype(int)
