#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np 
import tensorflow as tf
import os

MODEL_NAME="lstm_seq_model.ckpt"
MODEL_PATH=os.path.expanduser("~/TensorBoard/Models/")+MODEL_NAME

TRAIN_VIS_PATH=os.path.expanduser("~/TensorBoard/")

def generate_pattern(max_sequence=20, min_sequence=1,in_bits=10, out_bits=8, 
					pad=0.001, low_tol=0.001, high_tol=1.0):  # Function to generate sequences of different lengths

	seq_len_row = np.random.randint(low=min_sequence,high=max_sequence+1)

	pat = np.random.randint(low=0, high=2, size=(seq_len_row,out_bits))
	pat = pat.astype(np.float32)
	pat[pat < 1] = low_tol
	pat[pat >= 1] = high_tol

	x = np.ones(((max_sequence*2)+2,in_bits), dtype=pat.dtype) * pad
	y = np.ones(((max_sequence*2)+2,out_bits), dtype=pat.dtype) * pad # Side tracks are not produced

	# Creates a delayed output (Target delay)
	x[1:seq_len_row+1,2:] = pat
	y[seq_len_row+2:(2*seq_len_row)+2,:] = pat # No side tracks needed for the output (max_row+2)-seq_len_row

	x[1:seq_len_row+1,0:2] = low_tol
	x[0,:] = low_tol
	x[0,1] = 1.0 # Start of sequence
	x[seq_len_row+1,:] = low_tol
	x[seq_len_row+1,0] = 1.0 # End of sequence

	return x, y, pat

def create_training_data(no_of_examples=100,max_sequence=20, min_sequence=1,in_bits=10, out_bits=8, 
					pad=0.001, low_tol=0.001, high_tol=1.0):  # Function to generate the training data

	ti = []
	to = []

	for i in range(no_of_examples):
		x, y, _ = generate_pattern(max_sequence, min_sequence,in_bits, out_bits, 
									pad, low_tol, high_tol)

		ti.append(x)
		to.append(y) # The pattern with the two extra padded column is used

	return ti, to

seq_len = 20
bits=8 # The actual vector size for copying task
in_bits = bits+2 # The extra side track
out_bits = bits 

lr = 3e-5
m = 0.9 # Momentum
grad_clip=10
act_seq_len = (seq_len*2)+2 # Actual sequence lenght which includes the delimiters (Start and Stop bits on the side tracks)
no_hidden = [256,256,256] # No of LSTM layer and units per layer

# Defining the model
# Data and target place holder definition
# Data input is in [batch size, sequence length/time step, input dimension], batch major
data = tf.placeholder(dtype=tf.float32, shape=[None,act_seq_len,in_bits]) # Input
target = tf.placeholder(dtype=tf.float32, shape=[None,act_seq_len,out_bits]) # Ground truth

# A 3-layer LSTM
lstm_cells = [tf.nn.rnn_cell.LSTMCell(num_units=nh) for nh in no_hidden]
stacked_lstm_cells = tf.nn.rnn_cell.MultiRNNCell(cells=lstm_cells, state_is_tuple=True)
lstm_outputs, lstm_states = tf.nn.dynamic_rnn(cell=stacked_lstm_cells, inputs=data, initial_state=None, dtype=tf.float32)

rs = tf.reshape(lstm_outputs,[-1,no_hidden[-1]]) # The number of units in the last LSTM layer
dense_output = tf.layers.dense(inputs=rs,units=out_bits,activation=tf.nn.sigmoid,name="dense_output")
prediction = tf.reshape(dense_output,[-1,out_bits])
target_flat = tf.reshape(target,[-1,out_bits])
act_prediction = tf.reshape(prediction,[-1,act_seq_len,out_bits])

print("Total Parameters: {}".format(np.sum([np.prod(v.shape) for v in tf.trainable_variables()])))

# Binary cross entropy is used 
cross_entropy = -tf.reduce_mean((target_flat * tf.log(tf.clip_by_value(prediction,1e-6,0.9999)))+((1-target_flat) * tf.log(1-tf.clip_by_value(prediction,1e-10,0.9999))))
optimizer = tf.train.RMSPropOptimizer(learning_rate=lr,momentum=m)
grad_var_pair = optimizer.compute_gradients(loss=cross_entropy)
capped_grads_vars = [(tf.clip_by_value(grad, -grad_clip, grad_clip),var) for grad, var in grad_var_pair]
minimizer = optimizer.apply_gradients(grads_and_vars=capped_grads_vars)

# Mean Absolute Error is used as the error metric
error = tf.reduce_mean(tf.abs(tf.subtract(x=target_flat,y=prediction)))

init_op = tf.initializers.global_variables()

saver = tf.train.Saver()

train_summary = tf.summary.scalar("Training loss",cross_entropy)
val_summary = tf.summary.scalar("Validation loss",cross_entropy)

def train_lstm_seq():

	with tf.Session() as sess:
		sess.run(init_op)

		summary_writer = tf.summary.FileWriter(TRAIN_VIS_PATH, graph=tf.get_default_graph())

		batch_size = 100
		epoch = 1000
		stop = False

		print("Training")

		for i in range(epoch):
			try:	
				if stop:
					break

				for j in range(1000):
					t_inp, t_out = create_training_data(no_of_examples=batch_size,max_sequence=20, min_sequence=1,in_bits=10, out_bits=8)

					_, train_loss, train_error, ts = sess.run([minimizer,cross_entropy,error,train_summary],{data: t_inp, target: t_out})

					summary_writer.add_summary(ts,(i*1000)+j)
					
					print("Epoch: {:4d} | Train loss: {:8.4f} | Train error: {:8.4f}".format(i,train_loss,train_error))

					if train_loss <= 0.0001:
						print("Required training loss acheived.")
						print("Training loss: {} | Training error: {}".format(train_loss,train_error))
						stop = True
						break

			except KeyboardInterrupt:
				print("User interrupted")
				break

		print("Training done")

		print("Saving model")
		saver.save(sess,MODEL_PATH)
		print("Model saved")

		print("Testing")

		total_test_lss = 0.0
		total_test_err = 0.0

		for l in range(100):
			tst_inp, tst_out = create_training_data(no_of_examples=batch_size,max_sequence=20, min_sequence=1,in_bits=10, out_bits=8)

			lss, err = sess.run([cross_entropy,error],{data:tst_inp, target:tst_out})

			total_test_lss += lss
			total_test_err += err

		# The total error/loss is accumalated over 100 batches 
		test_lss = total_test_lss/100.0
		test_err = total_test_err/100.0

		print("Test loss: {:8.4f} | Test error: {:8.4f}".format(test_lss,test_err))
		print("Testing done")

def predictions_lstm_seq():

	with tf.Session() as sess:
		sess.run(init_op)

		print("Predicting")

		print("Restoring model")
		saver.restore(sess,MODEL_PATH)
		print("Model restored")
		
		x, _, _ = generate_pattern(max_sequence=20, min_sequence=1,in_bits=10, out_bits=8)
		
		y = sess.run(act_prediction,{data:[x]})

		plt.matshow(x)
		plt.matshow(y[0])
		plt.show()

if __name__ == '__main__':
	train_lstm_seq()
	predictions_lstm_seq()