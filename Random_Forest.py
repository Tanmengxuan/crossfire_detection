from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import os.path
from os import mkdir,environ
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import json
import matplotlib
import tensorflow as tf
import pickle
matplotlib.pyplot.ioff()

configCPU = tf.ConfigProto(
        device_count = {'GPU': 0}
    )

def loadConfig_RF(configPath_RF=None):
	if not configPath_RF:
		cfg = {

			"max_depth" : 5,	
			"n_estimators" : 50,	
			"min_samples_leaf" : 60
		}
		print "loaded default config"

	else:
		with open(configPath_RF, "r") as configFile:
			json_str = configFile.read()
			cfg = json.loads(json_str)
			print cfg

	global MAX_DEPTH
	global N_ESTIMATORS
	global MIN_SAMPLES_LEAF
	
	MAX_DEPTH = cfg["max_depth"]
	N_ESTIMATORS  = cfg["n_estimators"]
	MIN_SAMPLES_LEAF = cfg["min_samples_leaf"]
	return cfg

def count_sample(dis_test):
	attack = 0 
	no_attack = 0

	for i in dis_test:
		if i ==1:
			attack+=1
		else:
			no_attack+=1
	print ("Attack: "+str(attack) + " NON_Attack: "+str(no_attack))


def convert_onehot(data):

	tar = tf.one_hot(data, 2)
	sess = tf.Session(config = configCPU)
	tar = (sess.run(tar))
	return tar	

def vary_threshold(target_test, pred_test):
	result = []
	target_onehot = convert_onehot(target_test)
	
	thresholds = np.arange(0.01,1,0.01)
	
	for threshold in thresholds:
		dis = discrete(threshold, pred_test)
		pred_onehot = convert_onehot(dis)	
		
		pred_dict = {'1000decoys_rf': pred_onehot}
		#with open('1000decoys_rf_onlywarmup.pickle', 'wb') as f:
		#	pickle.dump(pred_dict, f)	

		precision,recall,f1,support = metrics.precision_recall_fscore_support(target_onehot,pred_onehot)
		printout =  "Threshold: (%f), Precision:  (%f,%f), Recall:  (%f,%f), F1: (%f,%f)"%(threshold, precision[0], precision[1], recall[0], recall[1], f1[0], f1[1])
		result.append(printout) 

	for row in result:
		print row 

def discrete(threshold, pred_proba_test):

	dis_test = []
	for prob in pred_proba_test:
		if prob >= threshold:
			dis = 1.0
			dis_test.append(dis)
		else:
			dis = 0.0
			dis_test.append(dis)

	dis_test = np.array(dis_test)

	return dis_test	

def evaluate(label_test, dis_test, pos_label):

	precision  = metrics.precision_score(label_test, dis_test, pos_label) 
	recall  = metrics.recall_score(label_test, dis_test, pos_label) 

	if precision==0.0 and recall ==0.0:
		f1 =0.0 
	else:
		f1 =2*((precision*recall)/(precision+recall)) 

	return precision, recall, f1
   

def random_forest(data_train, data_test,label_train, label_test):
    df_data_train = pd.DataFrame(data_train)
    df_label_train = pd.DataFrame(label_train)
    df_label_train.rename(columns = {0:'label'}, inplace = True)

       
    df_data_test = pd.DataFrame(data_test)
    df_label_test = pd.DataFrame(label_test)
    df_label_test.rename(columns = {0:'label'}, inplace = True)
        
    print ("data train shape is : [{}]".format(df_data_train.shape))
    print ("data test shape is : [{}]".format(df_data_test.shape))
    print ("data test label shape is : [{}]".format(df_label_test.shape))
    
    df_label_test.loc[df_label_test['label'] == 0, "attack"] = 0  
    df_label_test.loc[df_label_test['label'] != 0, "attack"] = 1
    target_test = df_label_test['attack']
    
    regr = RandomForestClassifier(max_depth=MAX_DEPTH,random_state=6,n_estimators=N_ESTIMATORS,min_samples_leaf=MIN_SAMPLES_LEAF,n_jobs=-1) 
    regr.fit(df_data_train, label_train)
    pred_test = regr.predict_proba(df_data_test)
    pred_test = np.delete(pred_test,0,1) 
   	

    return label_test, pred_test

def run_rf_once(trainx,trainy,testx,testy):
    trainy = np.array(trainy) #convert list to array
    testy = np.array(testy)
    
    
    target_test, pred_test = random_forest(trainx, testx,trainy, testy)
    vary_threshold(target_test, pred_test)
	
            
    print("area under curve (auc): ", metrics.roc_auc_score(target_test, pred_test))
     
    print("\n")
