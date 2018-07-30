import tensorflow as tf
import numpy as np
import pdb

def weighted_crossentropy(y_true, y_pred):
	output = y_pred
	#pdb.set_trace()
	output /= tf.reduce_sum(y_pred, -1, True)
	output = tf.clip_by_value(output, 1e-5, 1. - 1e-5)

	pos_weight = tf.cast(tf.reduce_sum(y_true[:, :, 0]), tf.float32) / (tf.reduce_sum(y_true[:, :, 1]) +1e-5)
	loss = y_true[:,:,1]*pos_weight*tf.log(output)[:,:,1] + y_true[:,:,0]*1*tf.log(output)[:,:,0]
	xent = - tf.reduce_mean(loss)
	return xent

