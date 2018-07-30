
import matplotlib 
matplotlib.use('Agg')
import input_data_old
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import keras
import tensorflow as tf
from keras import backend as k
import os.path
import numpy as np
import pandas as pd
from utils import weighted_crossentropy
from os import mkdir,environ
import json
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Activation, TimeDistributed, Bidirectional, Masking
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras import backend as K
import matplotlib.pyplot as plt
from collections import deque
from itertools import izip
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc, precision_score, recall_score, f1_score
import argparse
import time
import pickle

framework = "keras"
draw_graph =0 
run_on_cpu = False
doLoadConfig = -1 # set this to "1" when doing hyperparameter search

global PAD_VALUE
PAD_VALUE = 66.

global MAXLEN 
#MAXLEN = 597 
MAXLEN = 705 

global WINDOW_SPLIT
WINDOW_SPLIT = 2 

global INPUT_PATH_TEST
INPUT_PATH_TEST = "/home/cuc/CrossFire-Detect/CrossFire-Detect/models/data/trafficData/servers_1000/30_1000decoys_localmean/1000decoys_0_0_1_noise/normed*"

def loadConfig(configPath=None):
	if not configPath:
		cfg = {

			"input_path" : "/home/cuc/CrossFire-Detect/CrossFire-Detect/models/data/trafficData/servers_1000/30_1000decoys_localmean/normed*",


			"lstm_size" : 400,
			"n_inputs" : 64,
			"learning_rate" : 0.001,
			"beta_1" : 0.9,
			"beta_2" : 0.999,
			"weight_decay" : 0.01, 
			"num_features" : 996,
			"drop_rate" : 0.5,
			"drop_w" : 0.65,
			"drop_u" : 0.0,

			"epoch" : 20,
			"batch_size": 64 
		}
		print "loaded default config"

	else:
		with open(configPath, "r") as configFile:
			json_str = configFile.read()
			cfg = json.loads(json_str)
			print cfg

	global LSTM_SIZE
	global N_INPUTS
	global INPUT_PATH
	global LEARNING_RATE
	global BETA_1
	global BETA_2
	global WEIGHT_DECAY
	global NUM_FEATURES
	global DROP_RATE
	global DROP_W
	global DROP_U
	global EPOCH
	global BATCH_SIZE

	LSTM_SIZE = cfg["lstm_size"]
	N_INPUTS = cfg["n_inputs"]
	INPUT_PATH = cfg["input_path"]
	LEARNING_RATE = cfg["learning_rate"]
	BETA_1 = cfg["beta_1"]
	BETA_2 = cfg["beta_2"]
	WEIGHT_DECAY = cfg["weight_decay"]
	NUM_FEATURES = cfg["num_features"]
	DROP_RATE = cfg["drop_rate"]
	DROP_W = cfg["drop_w"]
	DROP_U = cfg["drop_u"]
	EPOCH = cfg["epoch"]
	BATCH_SIZE = cfg["batch_size"]

	return cfg

def getNextRunDir(prefix):
	script_path = os.path.dirname(os.path.realpath(__file__))

	output_path = os.path.join(script_path, "lstm_runs/servers_1000/30_1000decoys_localmean/",prefix)

	#Find the directory name in the series which is not used yet
	for num_run in range(0,500000):
		if not os.path.isdir(output_path+'_{}'.format(num_run)):
			mkdir(output_path+'_{}'.format(num_run))
			output_path = output_path+'_{}'.format(num_run)
			break
	return output_path

def f1(y_true, y_pred):
    #import pdb
    #pdb.set_trace()
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        #print y_true, y_pred
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

	y_true = np.reshape(y_true,(y_true.shape[0]*y_true.shape[1], y_true.shape[2]))
	y_pred = np.reshape(y_pred,(y_pred.shape[0]*y_pred.shape[1], y_pred.shape[2]))

	y_true = y_true[:,1]
	y_pred = y_pred[:,1]
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def get_result(test_pred, testY, threshold):
    
	pred = test_pred.reshape(-1, test_pred.shape[2])
	pred = pred > threshold
	pred = pred.astype(int)
	pred_pd = pd.DataFrame(pred)
	pred_pd.rename(columns = {0:"pred_nonattack", 1:"pred_attack"}, inplace = True)
	
	targ = testY.reshape( -1, testY.shape[2])
	targ_pd = pd.DataFrame(targ)
	targ_pd.rename(columns = {0:"targ_nonattack", 1:"targ_attack"}, inplace = True)
	
	combined_pd = pd.concat([pred_pd, targ_pd], axis=1, join_axes=[pred_pd.index])
	final_pd = combined_pd.drop(combined_pd[(combined_pd.targ_nonattack == 0.0) & (combined_pd.targ_attack == 0.0)].index)
	pred = final_pd.iloc[:, :2].copy()
	targ = final_pd.iloc[:, 2:].copy()
	_test_precision,_test_recall,_test_f1,_support = precision_recall_fscore_support(targ, pred)
	
	return _test_precision, _test_recall, _test_f1, _support 

class Metrics(keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs={}):
		#import pdb
		#pdb.set_trace()		
		val_predict = self.model.predict(self.validation_data[0])
		val_targ = self.validation_data[1]
		_val_precision,_val_recall,_val_f1,_support =  get_result(val_predict, val_targ, 0.5)

		print "- val_f1_anomaly: (%f)" % (_val_f1[1])

		return

def printConfig(dir,cfg):
	str = json.dumps(cfg)
	with open(os.path.join(dir,"netConfig.json"),"wb") as f:
		f.write(str)

parser = argparse.ArgumentParser()
parser.add_argument("--infer",help="Input the model path")
args = parser.parse_args()


if __name__ == "__main__":

	if not args.infer:    # training the Bilstm model
		configPath = None
		if doLoadConfig == 1:
			configPath = "curRunConfig_lstm.json"
		if run_on_cpu:
			os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
			os.environ["CUDA_VISIBLE_DEVICES"] = ""
		cfg = loadConfig(configPath)
	
		inp = input_data_old.inputter(INPUT_PATH, MAXLEN, PAD_VALUE, WINDOW_SPLIT)
		trainX, trainY = inp.getPadTrainData()
		print "train data shape", trainX.shape, trainY.shape
	
		inp = input_data_old.inputter(INPUT_PATH_TEST, MAXLEN, PAD_VALUE, WINDOW_SPLIT)
		validX, validY = inp.getPadValidationData()
		print "Validation data shape", validX.shape, validY.shape

		model = Sequential()
		model.add(Masking(mask_value= PAD_VALUE, input_shape=(MAXLEN, NUM_FEATURES)))
		model.add(Bidirectional(LSTM(LSTM_SIZE, return_sequences=True, dropout = DROP_W, recurrent_dropout = DROP_U, kernel_initializer='glorot_uniform'),))
		model.add(Bidirectional(LSTM(LSTM_SIZE, return_sequences=True, dropout = DROP_W, recurrent_dropout = DROP_U)))
		model.add(Bidirectional(LSTM(LSTM_SIZE, return_sequences=True, dropout = DROP_W, recurrent_dropout = DROP_U)))
		model.add(TimeDistributed(Dense(2, activation='softmax')))
		model.compile(loss= weighted_crossentropy, 
					  optimizer=keras.optimizers.Adam(lr=LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2, decay=WEIGHT_DECAY),
					  metrics=[f1])
		output_dir = getNextRunDir('bilstm_searchpad_001_multipledecoys')
		checkpoint = keras.callbacks.ModelCheckpoint(output_dir+'/best.h5',monitor= 'val_f1', mode = 'max', period = 1 , save_best_only=True)
		metrics = Metrics()
		history = model.fit(trainX, trainY, epochs=EPOCH, batch_size=BATCH_SIZE, validation_data =(validX, validY), verbose=1, shuffle = True,
			callbacks = [ metrics, checkpoint])
	
	
		print('Results stored in {}'.format(output_dir))
		printConfig(output_dir, cfg)
	
	
	if args.infer: # testing the Bilstm model
		start_time = time.time()
	
		model = load_model(args.infer, custom_objects={'weighted_crossentropy': weighted_crossentropy, 'f1':f1})
		inp = input_data_old.inputter(INPUT_PATH_TEST, MAXLEN, PAD_VALUE, WINDOW_SPLIT)
		testX, testY = inp.getPadTestData()
		test_predict = model.predict(testX)

		#pred_dict = {'testY': testY, "test_predict": test_predict}
		#with open('1000decoys_2bilstmpad_train.pickle', 'wb') as f:
		#	pickle.dump(pred_dict, f)	
		
		_test_precision,_test_recall,_test_f1,_support = get_result(test_predict, testY, 0.5)
	
		print " - test_f1: (%f,%f) - test_precision: (%f,%f) - test_recall (%f,%f) - test_support(%f,%f)" %(_test_f1[0],_test_f1[1], _test_precision[0],_test_precision[1], _test_recall[0],_test_recall[1], _support[0],_support[1])
		print "inference time:"
		print("--- %s seconds ---" % ((time.time() - start_time)))
		print("--- %s minutess ---" % ((time.time() - start_time)/60.0))
