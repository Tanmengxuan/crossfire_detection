import numpy as np
import pandas as pd
from sklearn import metrics
#import input_data
#import input_data_old
import input_data
import time
import pickle
import os.path
from os import mkdir,environ
import json
import argparse
import re
import Random_Forest
import Svm

def getNextRunDir(prefix):
	script_path = os.path.dirname(os.path.realpath(__file__))
	output_path = os.path.join(script_path, "botnet_runs/esNet_att30/",prefix)
	#Find the directory name in the series which is not used yet
	for num_run in range(0,500000):
		if not os.path.isdir(output_path+'_{}'.format(num_run)):
			mkdir(output_path+'_{}'.format(num_run))
			output_path = output_path+'_{}'.format(num_run)
			break
	return output_path

def printConfig_RF(dir,cfg):
	str = json.dumps(cfg)
	with open(os.path.join(dir,"netConfig_RF.json"),"wb") as f:
		f.write(str)

def printConfig_SVM(dir,cfg):
	str = json.dumps(cfg)
	with open(os.path.join(dir,"netConfig_SVM.json"),"wb") as f:
		f.write(str)

def get_label(trainY):
	
	trainY = np.reshape(trainY, (trainY.shape[0]*trainY.shape[1], trainY.shape[2]))
	trainY_reshape =[]
	for i in range(len(trainY)):
		if trainY[i][0]== 1:
			trainY_reshape.append(0.0)
		else:
			trainY_reshape.append(1.0)
	return trainY_reshape

def get_data_test(INPUT_PATH):

	#inp = input_data_old.inputter(INPUT_PATH)
	inp = input_data.inputter(INPUT_PATH)
	
	testX, testY = inp.getTestData(mergeAttackLabelsWithAtleast=False)
	testX_reshape = np.reshape(testX, (testX.shape[0]*testX.shape[1],testX.shape[2]*testX.shape[3]))
	del testX
	testY_reshape = get_label(testY)
	
	return testX_reshape, testY_reshape

def get_data_valid(INPUT_PATH):

	#inp = input_data_old.inputter(INPUT_PATH)
	inp = input_data.inputter(INPUT_PATH)
	
	validX, validY = inp.getValidationData(mergeAttackLabelsWithAtleast=False)
	validX_reshape = np.reshape(validX, (validX.shape[0]*validX.shape[1], validX.shape[2]*validX.shape[3]))
	del validX
	validY_reshape = get_label(validY) 

	return validX_reshape,validY_reshape

def get_data_train(INPUT_PATH):

	#inp = input_data_old.inputter(INPUT_PATH)
	inp = input_data.inputter(INPUT_PATH)
	
	trainX, trainY = inp.getTrainData(mergeAttackLabelsWithAtleast=False)
	trainX_reshape = np.reshape(trainX, (trainX.shape[0]*trainX.shape[1], trainX.shape[2]*trainX.shape[3]))
	del trainX
	trainY_reshape = get_label(trainY) 

	return trainX_reshape,trainY_reshape

start_time = time.time()

#INPUT_PATH ='data/trafficData/esNet_att30*'
#trainx,trainy,validx,validy,testx,testy = get_data(INPUT_PATH)

if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument("--infer_rf",help="Input the model path")
	parser.add_argument("--infer_svm",help="Input the model path")

	parser.add_argument("--input_train",help="path of input file")
	parser.add_argument("--input_test",help="path of input file")
	parser.add_argument("--input_valid",help="path of input file")

	parser.add_argument("--grid_search",help="change doLoadConfig to 1", action = "store_true")

	parser.add_argument("--train_rf", help="if random forest model is used", action = "store_true")
	parser.add_argument("--train_svm", help="if svm  model is used", action = "store_true")

	args = parser.parse_args()

	if args.input_train:
		INPUT_PATH = args.input_train
		trainx,trainy = get_data_train(INPUT_PATH)

	if args.input_valid:
		INPUT_PATH = args.input_valid
		validx,validy = get_data_valid(INPUT_PATH)

	if args.input_test:
		INPUT_PATH = args.input_test
		testx,testy = get_data_test(INPUT_PATH)

	if args.grid_search:
		doLoadConfig = 1
	else:
		doLoadConfig = 0

	if not (args.infer_rf and args.infer_svm):
		if args.train_rf:
			configPath_RF = None

			if doLoadConfig == 1:
				configPath_RF = "curRunConfig_RF.json"

			output_dir = getNextRunDir('rf_search')

			cfg_RF = Random_Forest.loadConfig_RF(configPath_RF)

			Random_Forest.run_rf_once(trainx,trainy,validx,validy)
			#Random_Forest.run_rf_once(trainx,trainy, testx,testy)

			printConfig_RF(output_dir,cfg_RF)
			print('Configs stored in {}'.format(output_dir))

		elif args.train_svm:
			configPath_SVM = None

			if doLoadConfig == 1:
				configPath_SVM = "curRunConfig_SVM.json"

			output_dir = getNextRunDir('svm_search')

			cfg_SVM = Svm.loadConfig_SVM_poly(configPath_SVM)

			Svm.run_svm_once(trainx,trainy,validx,validy)

			printConfig_SVM(output_dir,cfg_SVM)
			print('Configs stored in {}'.format(output_dir))

	if args.infer_rf:

		configPath_RF = os.path.join(args.infer_rf,'netConfig_RF.json')

		cfg_RF = Random_Forest.loadConfig_RF(configPath_RF)

		Random_Forest.run_rf_once(trainx,trainy,testx,testy)

	if args.infer_svm:

		configPath_SVM = os.path.join(args.infer_svm,'netConfig_SVM.json')

		cfg_SVM = Svm.loadConfig_SVM_poly(configPath_SVM)

		Svm.run_svm_once(trainx,trainy,testx,testy)
