#External imports
import numpy as np
import os
import pickle
import tensorflow as tf
from tensorflow_addons.losses import GIoULoss

from tensorflow import keras
from keras import Model
from keras.layers import Concatenate, Dense, Input, Lambda, LSTM, LSTMCell, Reshape, RNN
from keras.losses import MeanSquaredError, MeanAbsoluteError
from keras.metrics import Precision, Recall
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model

#Internal imports
from global_config.global_config import (
	ORIGINAL_IMAGE_HEIGHT, ORIGINAL_IMAGE_WIDTH,
	MODEL_PATH,
	N_INPUT_TSTEPS, N_OUTPUT_TSTEPS, NUM_CAMS, NUM_FEATURES,
	EPOCHS, TRAIN_BATCH_SIZE, TEST_BATCH_SIZE,
	CAM_LOSS, CAM_LOSS_WT)

from loader.loader import load_dataset
from utils.utils import (
	SIoU,
	store_pkl, load_pkl,
	tensor_encode_one_hot, tensor_decode_one_hot,
	get_partition, generate_targets, targets2tensors)
'''
CAVEAT: Using n + 1 cameras
'''


def custom_regression_loss(box_true, box_pred):
	box_loss = 0.01 * MeanSquaredError()(box_true, box_pred)
	diag_loss = 0.01 * MeanAbsoluteError()(
		    tf.square(box_true[:, 2]) + tf.square(box_true[:, 3]),
			tf.square(box_pred[:, 2]) + tf.square(box_pred[:, 3]))

	return box_loss + diag_loss


# Retained for compatibility
def combined_loss_fn(Y, Y_pred, num_cams=NUM_CAMS):
	cam_loss = tf.keras.losses.CategoricalCrossentropy()	# ~ 2.0
	box_loss = GIoULoss()						# < 2.0
	agg_loss = 4 * cam_loss(Y[:, :, 0: num_cams], Y_pred[:, :, 0: num_cams]) + \
		10 * box_loss(Y[:, :, num_cams:], Y_pred[:, :, num_cams:])
	return(agg_loss)


def save_model(model, logs=None, name="custom_lstm.ml", path=MODEL_PATH):
	model.save(os.path.join(path, name))
	if logs is not None:
		store_pkl(logs, os.path.join(path, name, "logs.pkl"))


def load_model(name, path=MODEL_PATH):
	model = tf.keras.models.load_model(os.path.join(path, name), \
			custom_objects={'custom_regression_loss': custom_regression_loss}) # combined_loss_fn': combined_loss_fn}) [for compatibility]
	try:
		logs = load_pkl(os.path.join(path, name, "logs.pkl"))
		return model, logs
	except:
		print(f"Could not find model logs.")
		return model, None


def define_model(n_input_tsteps=N_INPUT_TSTEPS, n_output_tsteps=N_OUTPUT_TSTEPS, num_features=NUM_FEATURES, num_cams=NUM_CAMS):
	"""Define the model architecture"""

	print("Building model...")
	predictions = []
	input_layer = Input(shape=(n_input_tsteps, num_features))
	# Prepare the context for the regressor in center-point format
	last_pos = Lambda(lambda x: x[:, -1, 1:])(input_layer)

	# Encoder LSTM
	encoder_lstm = LSTM(units=num_features, return_state=True)
	encoder_layer, h, c = encoder_lstm(input_layer)
	x = Reshape((1, num_features))(encoder_layer) # [!] VERY IMPORTANT TO INCLUDE A TIME-STEP

	# Decoder LSTM
	for block_index in range(n_output_tsteps):
		# LSTM cell
		y, h, c = RNN(LSTMCell(units=num_features), \
					  return_state=True)(x, initial_state=[h, c])
		intermediate_layer = Dense(units=num_features, activation='sigmoid')(y)

		# Prediction module
		classifier = Dense(units=num_cams, activation='softmax', name=f"classifier_{block_index}")
		cam_pred = classifier(intermediate_layer)

		regressor = Dense(units=4, activation='linear', kernel_regularizer=l2(0.01), name=f"regressor_{block_index}")
		context_layer = Concatenate(axis=-1)([last_pos, intermediate_layer])
		box_pred = regressor(context_layer)

		# Derive variables for the next timestep
		x = Reshape((1, num_features))(y) # [!] VERY IMPORTANT TO INCLUDE A TIME-STEP
		# last_pos = box_pred

		# Forward output
		predictions.append(cam_pred)
		predictions.append(box_pred)

	# Compile model
	loss = {}
	loss_weights = {}
	for i in range(n_output_tsteps):
		loss[f"classifier_{i}"] = CAM_LOSS
		loss_weights[f"classifier_{i}"] = CAM_LOSS_WT
		loss[f"regressor_{i}"] = custom_regression_loss
		loss_weights[f"regressor_{i}"] = 1

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
	# Uncomment the tensorboard call-back if you want diagnostics
	# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(MODEL_PATH, "debug_mse"), histogram_freq=1)
	logs = model.fit(X_train, targets_train, batch_size=train_batch_size, epochs=epochs, verbose=2) #\
    #       , callbacks=[tensorboard_callback])
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

def predict(model, X, probabilities=False):
	print("Generating predictions...")
	predictions = model.predict(X)
	y_pred = targets2tensors(predictions)
	if probabilities:
		return y_pred.numpy()
	else:
	    return tensor_decode_one_hot(y_pred).numpy().astype(int)

def calculate_metrics(model, X, Y, num_cams=NUM_CAMS, num_features=NUM_FEATURES):
	Y_pred_encoded = predict(model, X, probabilities=True)

	# Ensure that we are working with numpy arrays
	num_samples = Y.shape[0] * Y.shape[1]
	Y_pred = tensor_decode_one_hot(Y_pred_encoded)
	Y, Y_pred, Y_pred_encoded = np.array(Y), np.array(Y_pred), np.array(Y_pred_encoded)


	# Modify the output tensors for generating metrics
	cams_true = tf.one_hot(Y[:, :, 0].flatten(), depth=num_cams)
	cams_pred = Y_pred_encoded[:, :, :num_cams].reshape(num_samples, num_cams)

	parts_true = Y[:, :, 1:].reshape(num_samples, 4)
	parts_pred = Y_pred_encoded[:, :, num_cams:].reshape(num_samples, 4)
	# Clip the prediction co-ordinates
	for i in [0, 2]:
		parts_pred[:, i] = np.clip(parts_pred[:, i], 0, ORIGINAL_IMAGE_WIDTH - 1)
		parts_true[:, i] = np.clip(parts_true[:, i], 0, ORIGINAL_IMAGE_WIDTH - 1)
	for i in [1, 3]:
		parts_pred[:, i] = np.clip(parts_pred[:, i], 0, ORIGINAL_IMAGE_HEIGHT - 1)
		parts_true[:, i] = np.clip(parts_true[:, i], 0, ORIGINAL_IMAGE_HEIGHT - 1)
	parts_true = tf.one_hot(np.apply_along_axis( \
		lambda box_coords: get_partition(*box_coords), axis=1, arr=parts_true), \
			depth=25)
	parts_pred = tf.one_hot(np.apply_along_axis( \
		lambda box_coords: get_partition(*box_coords), axis=1, arr=parts_pred), \
		    depth=25)

	# Get the metrics
	siou_score = SIoU(Y.reshape(num_samples, num_features), \
					Y_pred.reshape(num_samples, num_features))

	avg_precision = Precision()

	avg_precision.update_state(cams_true, cams_pred)
	AP_cams = avg_precision.result()
	avg_precision.reset_state()

	avg_precision.update_state(parts_true, parts_pred)
	AP_parts = avg_precision.result()
	avg_precision.reset_state()

	# Obtain PR-curve stats
	thresholds=np.linspace(0.01, 0.99, 100).tolist()
	avg_precision = Precision(thresholds=thresholds)
	avg_recall = Recall(thresholds=thresholds)

	PR_cams = {}
	avg_precision.update_state(cams_true, cams_pred)
	PR_cams["precision"] = avg_precision.result()
	avg_precision.reset_state()

	avg_recall.update_state(cams_true, cams_pred)
	PR_cams["recall"] = avg_recall.result()
	avg_recall.reset_state()

	PR_parts = {}
	avg_precision.update_state(parts_true, parts_pred)
	PR_parts["precision"] = avg_precision.result()
	avg_precision.reset_state()

	avg_recall.update_state(parts_true, parts_pred)
	PR_parts["recall"] = avg_recall.result()
	avg_recall.reset_state()

	return AP_cams, AP_parts, siou_score, PR_cams, PR_parts
