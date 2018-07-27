#if __name__ == "__main__":

import matplotlib 
matplotlib.use('Agg')
import input_data_old
import input_data
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import keras
import tensorflow as tf
from keras import backend as k
import os.path
import numpy as np
import pandas as pd
from utils import weighted_crossentropy, weighted_categorical_crossentropy
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

#INPUT_PATH_TEST = "/home/cuc/CrossFire-Detect/CrossFire-Detect/models/data/trafficData/servers_100/50_100decoys_localmean/100decoys_0_0_1_noise/normed*"
#INPUT_PATH_TEST = "/home/cuc/CrossFire-Detect/CrossFire-Detect/models/data/trafficData/servers_100/50_100decoys_localmean/normed_before_split/normed*"

def loadConfig(configPath=None):
	if not configPath:
		cfg = {

			"input_path" : "/home/cuc/CrossFire-Detect/CrossFire-Detect/models/data/trafficData/servers_1000/30_1000decoys_localmean/normed*",
			#"input_path" : "/home/cuc/CrossFire-Detect/CrossFire-Detect/models/data/trafficData/servers_100/50_100decoys_localmean/100decoys_0_0_1_noise/normed*",
			#"input_path" : "/home/cuc/CrossFire-Detect/CrossFire-Detect/models/data/trafficData/servers_100/50_100decoys_localmean/normed_100decoys_0_0_1_noise_w64o0_train.csv",
			#"input_path" : "/home/cuc/CrossFire-Detect/CrossFire-Detect/models/data/trafficData/servers_100/50_100decoys_localmean/normed_before_split/normed*",


			#"input_path" : "../data/lstm_esNet_w64o16*",
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

def precision(y_true, y_pred):
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	return precision

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

	#y_pred = y_pred[:,:,1]
	#y_true = y_true[:,:,1a_2=BETA_2, decay=WEIGHT_DECAY]
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
	def on_train_begin(self, logs={}):
		self.val_f1s = []
		self.val_recalls = []
		self.val_precisions = []
		self.anomaly_f1 = []

#	def on_train_end(self, logs={}):
#
#		train_predict = (np.asarray(self.model.predict(self.trainX))).round()
#		train_predict = train_predict.reshape(train_predict.shape[0]*train_predict.shape[1], train_predict.shape[2])
#		train_targ = self.trainY
#		train_targ = train_targ.reshape(train_targ.shape[0]*train_targ.shape[1], train_targ.shape[2])
#		train_pred_anomaly = train_predict[:,1]
#		recall_anomaly  = recall_score(train_targ_anomaly, train_pred_anomaly, pos_label=1)
#		f1_anomaly =2*((precision_anomaly*recall_anomaly)/(precision_anomaly+recall_anomaly)) 
#		self.anomaly_f1.append(f1_anomaly)
#		print "- train_f1_anomaly: (%f)" % (f1_anomaly)
#
	def on_epoch_end(self, epoch, logs={}):
		#val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
		#val_predict = (np.asarray(self.model.predict(self.validation_data[0])))
		#val_predict = np.where(np.isnan(val_predict), 0, val_predict)
		#val_predict = np.clip(val_predict, 1e-5, 1. - 1e-5)
		#val_predict = val_predict > 0.5
		#val_predict = val_predict.astype(int)
		#val_predict = val_predict.reshape(val_predict.shape[0]*val_predict.shape[1], val_predict.shape[2])
		#val_targ = self.validation_data[1]
		#val_targ = val_targ.reshape(val_targ.shape[0]*val_targ.shape[1], val_targ.shape[2])
		#_val_precision,_val_recall,_val_f1,_support = precision_recall_fscore_support(val_targ,val_predict)
		#import pdb
		#pdb.set_trace()		
		val_predict = self.model.predict(self.validation_data[0])
		val_targ = self.validation_data[1]
		_val_precision,_val_recall,_val_f1,_support =  get_result(val_predict, val_targ, 0.5)

		#val_pred_anomaly = val_predict[:,1]
		#val_targ_anomaly = val_targ[:,1]
		#precision_anomaly  = precision_score(val_targ_anomaly, val_pred_anomaly, pos_label=1)
		#recall_anomaly  = recall_score(val_targ_anomaly, val_pred_anomaly, pos_label=1)
		#try:
		#	f1_anomaly = f1_score(val_targ_anomaly, val_pred_anomaly, pos_label=1)
		#except:
		#	f1_anomaly = 0.0
		#f1_anomaly =2*((precision_anomaly*recall_anomaly)/(precision_anomaly+recall_anomaly)) 
		#self.anomaly_f1.append(f1_anomaly)
		print "- val_f1_anomaly: (%f)" % (_val_f1[1])

		#try:
		#	val_predict = (np.asarray(self.model.predict(self.validation_data[0])))
		#	val_predict = val_predict.reshape(val_predict.shape[0]*val_predict.shape[1], val_predict.shape[2])
		#	val_predict = np.clip(val_predict, 1e-5, 1. - 1e-5)
		#	pos_weight = np.sum(val_targ[:,0]) / np.sum(val_targ[:,1])
		#	loss = val_targ[:,1]*pos_weight*np.log(val_predict[:,1]) + val_targ[:,0]*1*np.log(val_predict[:,0])
		#	xent = -np.mean(loss)
		#except:
		#	xent = 0.0
		#print "- val_weighted_loss: (%f)" % (xent)
		#loss = np.sum(val_targ*np.log(val_predict) + (1- val_targ)*np.log(val_predict))/(-1*103168)
		#keras_loss = K.categorical_crossentropy(K.variable(val_predict), K.variable(val_targ))
		#keras_loss = np.sum(K.eval(keras_loss))/103168
		#print "- val_LOSS: (%f)" %(loss)
		#print "- val_KerasLOSS:{} ".format(keras_loss)

		#val_predict = self.model.predict(self.validation_data[0])
		#val_targ = self.validation_data[1]
		#keras_loss = K.categorical_crossentropy(K.variable(val_predict), K.variable(val_targ))
		#keras_loss = np.sum(K.eval(keras_loss))/103168
		#print "- val_KerasLOSS:{} ".format(keras_loss)

		#self.val_f1s.append(_val_f1)
		#self.val_recalls.append(_val_recall)
		#self.val_precisions.append(_val_precision)
		#print " - val_f1: (%f,%f) - val_precision: (%f,%f) - val_recall (%f,%f) - val_support(%f,%f)" %(_val_f1[0],_val_f1[1], _val_precision[0],_val_precision[1], _val_recall[0],_val_recall[1], _support[0],_support[1])
		return

def printConfig(dir,cfg):
	str = json.dumps(cfg)
	with open(os.path.join(dir,"netConfig.json"),"wb") as f:
		f.write(str)

def findBestWindow(preds,labels):
	buffer_length = 7
	atleast =7
	pred_buffer = deque(maxlen = buffer_length)
	detections = 0
	first_detection_sample=[] # the point at which all 7 values in buffer are predicted attacks
	misses = 0
	attack_sample_no = 0
	missed_evenafter = []
	false_positives = 0
	false_in_this_window =0
	for window, window_l in izip(preds, labels):
		if attack_sample_no !=0: # last attack sample number in the previous window
			misses +=1
			missed_evenafter.append(attack_sample_no)
		false_positives += false_in_this_window
		false_in_this_window = 0
		for _ in range(buffer_length): #creating a new buffer for the new window
			pred_buffer.append(0)
		attack_sample_no = 0
		for val,val_l in izip(window,window_l):
			pred_buffer.append(val[1])
			if val_l[1] == 1:
				if sum(pred_buffer) >= atleast: # if all 7 values in buffer are attacks
					#Correct match
					detections +=1
					print attack_sample_no
					first_detection_sample.append(attack_sample_no)
					attack_sample_no =0
					break  # escape for loop and examine a new window line 102, each window has only one sequence of attack
				else:
					attack_sample_no +=1 # attack sample number in a sequence of attack
			if attack_sample_no == 0 and sum(pred_buffer) >= atleast: # no actual attack & 7 predicted attack
				false_in_this_window = 1 # flag as false positive
				print window[:,1], window_l[:,1]
	print first_detection_sample
	average_detection_sample = sum(first_detection_sample) / float(len(first_detection_sample)+0.1)
	print "detections: {}, misses {}, avg {}".format(detections, misses,average_detection_sample)
	actual_misses = [i for i in missed_evenafter if i >average_detection_sample] # if attack sample number greater than latency number.
	cant_blame_misses = [i for i in missed_evenafter if i<=average_detection_sample]
	print "actual_misses {}, cant_blame_misses {}".format(len(actual_misses), len(cant_blame_misses))
	print "false_p sitives {}".format(false_positives)
	print "precision {}, recall {}".format(detections/float(detections+false_positives), detections/float(detections+len(actual_misses)))


parser = argparse.ArgumentParser()
parser.add_argument("--infer",help="Input the model path")
args = parser.parse_args()


if __name__ == "__main__":

	if not args.infer:
		configPath = None
		if doLoadConfig == 1:
			configPath = "curRunConfig_lstm.json"
		if run_on_cpu:
			os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
			os.environ["CUDA_VISIBLE_DEVICES"] = ""
		cfg = loadConfig(configPath)
	
		#inp = input_data_old.inputter(INPUT_PATH)
		inp = input_data_old.inputter(INPUT_PATH, MAXLEN, PAD_VALUE, WINDOW_SPLIT)
		#trainX, trainY = inp.getTrainData(mergeAttackLabelsWithAtleast=False)
		trainX, trainY = inp.getPadTrainData()
		#trainX = trainX.reshape(trainX.shape[0],trainX.shape[1],trainX.shape[2])
		print "train data shape", trainX.shape, trainY.shape
	
		#inp = input_data_old.inputter(INPUT_PATH_TEST)
		inp = input_data_old.inputter(INPUT_PATH_TEST, MAXLEN, PAD_VALUE, WINDOW_SPLIT)
		#validX, validY = inp.getValidationData(mergeAttackLabelsWithAtleast=False)
		validX, validY = inp.getPadValidationData()
		#validX = validX.reshape(validX.shape[0],validX.shape[1],validX.shape[2])
		print "Validation data shape", validX.shape, validY.shape

		#trainX = np.concatenate((trainX, validX), axis=0)
		#trainY = np.concatenate((trainY, validY), axis=0)
	
		#print "train data shape", trainX.shape, trainY.shape
		model = Sequential()
		model.add(Masking(mask_value= PAD_VALUE, input_shape=(MAXLEN, NUM_FEATURES)))
		model.add(Bidirectional(LSTM(LSTM_SIZE, return_sequences=True, dropout = DROP_W, recurrent_dropout = DROP_U, kernel_initializer='glorot_uniform'),))
		model.add(Bidirectional(LSTM(LSTM_SIZE, return_sequences=True, dropout = DROP_W, recurrent_dropout = DROP_U)))
		model.add(Bidirectional(LSTM(LSTM_SIZE, return_sequences=True, dropout = DROP_W, recurrent_dropout = DROP_U)))
		#model.add(LSTM(LSTM_SIZE, return_sequences=True, input_shape=(N_INPUTS,NUM_FEATURES), dropout = DROP_RATE))
		#model.add(LSTM(LSTM_SIZE, return_sequences=True, dropout = DROP_RATE))
		model.add(TimeDistributed(Dense(2, activation='softmax')))
		model.compile(loss= weighted_crossentropy, 
					  optimizer=keras.optimizers.Adam(lr=LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2, decay=WEIGHT_DECAY),
					  metrics=[f1])
		#early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')
		#path = "lstm_runs/servers_1000/30_1000decoys_localmean/bilstm_searchpad_001_1000decoys_14/best.h5"
		#model = load_model(path, custom_objects={'weighted_crossentropy': weighted_crossentropy, 'f1':f1}) 
		output_dir = getNextRunDir('bilstm_searchpad_001_multipledecoys')
		#tbCallBack = keras.callbacks.TensorBoard(log_dir=output_dir, histogram_freq=1, write_grads=True, write_graph= False, write_images=True)
		checkpoint = keras.callbacks.ModelCheckpoint(output_dir+'/best.h5',monitor= 'val_f1', mode = 'max', period = 1 , save_best_only=True)
		metrics = Metrics()
		history = model.fit(trainX, trainY, epochs=EPOCH, batch_size=BATCH_SIZE, validation_data =(validX, validY), verbose=1, shuffle = True,
			callbacks = [ metrics, checkpoint])
	
		#testX, testY = inp.getTestData(mergeAttackLabelsWithAtleast=False)
		#testX = testX.reshape(testX.shape[0],testX.shape[1],testX.shape[2])
		#test_predict = (np.asarray(model.predict(testX))).round()
	
		#_test_precision,_test_recall,_test_f1,_support = precision_recall_fscore_support(np.reshape(testY,(testY.shape[0]*testY.shape[1], testY.shape[2])),np.reshape(test_predict, (test_predict.shape[0]*test_predict.shape[1], test_predict.shape[2])))
	
		#print " - test_f1: (%f,%f) - test_precision: (%f,%f) - test_recall (%f,%f) - test_support(%f,%f)" %(_test_f1[0],_test_f1[1], _test_precision[0],_test_precision[1], _test_recall[0],_test_recall[1], _support[0],_support[1])
	
		if draw_graph:
			y_score = model.predict_proba(validX)
			prob_1 = y_score.reshape(-1,2)[:,1]
			label_1 = validY.reshape(-1,2)[:,1]
			fpr, tpr, _ = roc_curve(label_1, prob_1)
			roc_auc = auc(fpr, tpr)
			print roc_auc
			plt.figure()
			lw = 2
			plt.plot(fpr, tpr, color='darkorange',
					 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
			plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
			plt.xlim([0.0, 1.0])
			plt.ylim([0.0, 1.05])
			plt.xlabel('False Positive Rate')
			plt.ylabel('True Positive Rate')
			plt.title('LSTM ROC')
			plt.legend(loc="lower right")
			plt.savefig("LSTM")
		val_predict = (np.asarray(model.predict(validX))).round()
		print('Results stored in {}'.format(output_dir))
		#print val_predict
		#findBestWindow(val_predict,validY)
		printConfig(output_dir, cfg)
	
		#print('Results stored in {}'.format(output_dir))
		#if draw_graph:
			#plt.savefig("LSTM2")
	
	
	if args.infer:
		start_time = time.time()
	
		model = load_model(args.infer, custom_objects={'weighted_crossentropy': weighted_crossentropy, 'f1':f1})
		#INPUT_PATH_TRAIN =  "/home/cuc/CrossFire-Detect/CrossFire-Detect/models/data/trafficData/servers_100/50_100decoys_localmean/normed_100decoys_0_0_1_noise_w64o0_train.csv"
		#inp = input_data_old.inputter("/home/cuc/CrossFire-Detect/CrossFire-Detect/models/data/trafficData/servers_100/50_100decoys/100decoys_0_0_1_noise/normed*")
		#inp = input_data.inputter("../data/lstm_esNet_w64o16_t*")
		inp = input_data_old.inputter(INPUT_PATH_TEST, MAXLEN, PAD_VALUE, WINDOW_SPLIT)
		#inp = input_data.inputter(INPUT_PATH_TEST)
		#testX, testY = inp.getTestData(mergeAttackLabelsWithAtleast=False)
		testX, testY = inp.getPadTestData()
		#testX = testX.reshape(testX.shape[0],testX.shape[1],testX.shape[2])
		#test_predict = (np.asarray(model.predict(testX))).round()
		#test_predict = (np.asarray(model.predict(testX)))
		#test_predict = test_predict > 0.4
		#test_predict = test_predict.astype(int)
		#_test_precision,_test_recall,_test_f1,_support = precision_recall_fscore_support(np.reshape(testY,(testY.shape[0]*testY.shape[1], testY.shape[2])),np.reshape(test_predict, (test_predict.shape[0]*test_predict.shape[1], test_predict.shape[2])))
		test_predict = model.predict(testX)
		pred_dict = {'testY': testY, "test_predict": test_predict}
		with open('1000decoys_2bilstmpad_train.pickle', 'wb') as f:
			pickle.dump(pred_dict, f)	
		
		_test_precision,_test_recall,_test_f1,_support = get_result(test_predict, testY, 0.5)
	
		print " - test_f1: (%f,%f) - test_precision: (%f,%f) - test_recall (%f,%f) - test_support(%f,%f)" %(_test_f1[0],_test_f1[1], _test_precision[0],_test_precision[1], _test_recall[0],_test_recall[1], _support[0],_support[1])
		print "inference time:"
		print("--- %s seconds ---" % ((time.time() - start_time)))
		print("--- %s minutess ---" % ((time.time() - start_time)/60.0))
		#findBestWindow(val_predict,validY)
