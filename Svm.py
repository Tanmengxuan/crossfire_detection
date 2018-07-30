from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn import metrics
import os.path
from os import mkdir,environ
import json
import time

def loadConfig_SVM_rbf(configPath_SVM=None):
	if not configPath_SVM:
		cfg = {

			"gamma" : 0.1,	
			#"n_estimators" : 50,	
			"c" : 60
		}
		print "loaded default config"

	else:
		with open(configPath_SVM, "r") as configFile:
			json_str = configFile.read()
			cfg = json.loads(json_str)
			print cfg

	global GAMMA
	#global N_ESTIMATORS
	global C
	
	GAMMA = cfg["gamma"]
	#N_ESTIMATORS  = cfg["n_estimators"]
	C = cfg["c"]
	return cfg

def loadConfig_SVM_poly(configPath_SVM=None):
	if not configPath_SVM:
		cfg = {

			"degree" : 3,	
			"coef0" : 1,	
			"c" : 5
		}
		print "loaded default config"

	else:
		with open(configPath_SVM, "r") as configFile:
			json_str = configFile.read()
			cfg = json.loads(json_str)
			print cfg

	global DEGREE
	global COEF0
	global C
	
	DEGREE = cfg["degree"]
	COEF0  = cfg["coef0"]
	C = cfg["c"]
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


def vary_threshold(label_test, pred_proba_test, min_pred, max_pred):
	
	thresholds = np.arange(min_pred,max_pred,0.01)
	dis_test = []

	for i in thresholds:
		threshold = i

		for prob in pred_proba_test:
			if prob >= threshold:
				dis = 1.0
				dis_test.append(dis)
			else:
				dis = 0.0
				dis_test.append(dis)
		count_sample(dis_test)
		dis_test = np.array(dis_test)

		precision  = metrics.precision_score(label_test, dis_test, pos_label=1) 
		recall  = metrics.recall_score(label_test, dis_test, pos_label=1) 
		if precision==0.0 and recall ==0.0 :
			f1 =0.0 
		else:
			f1 =2*((precision*recall)/(precision+recall)) 
		print ("Threshold: "+str(threshold)+", Precision: "+str(precision)+", Recall: "+str(recall)+", F1: "+str(f1))
		print ("\n")
		dis_test = []

def svm_svc_rbf(data_train, data_test,label_train, label_test):

	svm_clf_rbf = Pipeline([
	("scaler", StandardScaler()),
	("svm_clf", SVC(kernel="rbf", gamma=GAMMA, C=C, random_state = 6))
	])

	svm_clf_rbf.fit(data_train, label_train)

	pred_test = svm_clf_rbf.decision_function(data_test)
	
	return label_test, pred_test

def svm_svc_poly(data_train, data_test,label_train, label_test):

	print "hyper"
	print DEGREE
	print COEF0
	print C

	svm_clf_poly = Pipeline([
	("scaler", StandardScaler()),
	("svm_clf", SVC(kernel="poly", degree=DEGREE,coef0=COEF0, C=C, random_state = 6))
	])
	
	start_time = time.time()
	svm_clf_poly.fit(data_train, label_train)
	print("--- %s minutess ---" % ((time.time() - start_time)/60.0))
	pred_test = svm_clf_poly.decision_function(data_test)

	return label_test, pred_test


def run_svm_once(trainx,trainy,testx,testy):
	trainy = np.array(trainy) #convert list to array
	testy = np.array(testy)
    
    
	target_test, pred_test = svm_svc_poly(trainx, testx,trainy, testy)
	max_pred = np.max(pred_test)
	print "max_pred: " + str(max_pred)
	min_pred = np.min(pred_test)
	print "min_pred: " + str(min_pred)
	vary_threshold(target_test, pred_test, min_pred, max_pred) 
	print("area under curve (auc): ", metrics.roc_auc_score(target_test, pred_test)) 
