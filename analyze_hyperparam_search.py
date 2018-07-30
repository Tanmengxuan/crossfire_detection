import os.path
import glob
import argparse
import json
import re
import warnings
import csv

script_dir = os.path.dirname(os.path.realpath(__file__))

def analyzeRunsDataConvnet(prefix, path, csvpath):
	dirs = glob.glob(os.path.join(path,prefix)+"*")
	if not dirs:
		raise ValueError("Input path and prefix didn't match any directory. Check input params")
	print "Number of directories found {}".format(len(dirs))
	outFile = open(csvpath,"wb")
	csvFile = csv.writer(outFile, delimiter='\t')
	csvFile.writerow(["run", "best_epoch", "accuracy","val_loss", "precision_0","precision_1", "recall_0", "recall_1",\
		"layer1_stride", "layer2_stride_width", "filter1_height", "filter1_width", "layer2_stride_height",\
		 "num_features", "batch_size", "filter2_width", "window_size", "filter1_depth", "drop_rate",\
		  "num_epochs", "filter2_depth", "num_classes", "filter2_height","test_acc","test_loss","test_pre_0","test_pre_1","test_rec_0","test_rec_1"])
	for adir in dirs:
		acc_path = os.path.join(adir, "accuracy.txt")
		if not os.path.exists(acc_path):
			continue
		with open(acc_path , "r") as acc_file:
			acc_str = acc_file.read()
		acc_str = acc_str.replace("^H","")
		regex_str = r"epoch\s*(\d+).*val\sacc:\s*(\d+\.\d+).*val\sloss:\s*(\d+\.\d+).*prec:\s*\((\d+\.\d+),\s*(\d+\.\d+)\).*rec:\s*\((\d+\.\d+),\s*(\d+\.\d+)\)"
		regex = re.compile(regex_str)
		#ret = re.findall(regex, acc_str)
		match = re.search(regex,acc_str)
		if not match:
			warnings.warn("accuracy.txt regex did not match for dir : {},skipping".format(adir))
			continue
		epoch = match.group(1)
		accuracy = match.group(2)
		loss_val = match.group(3)
		precision_0 = match.group(4)
		precision_1 = match.group(5)
		recall_0 = match.group(6)
		recall_1 = match.group(7)

		out_path = os.path.join(adir, "output.txt")
		if not os.path.exists(out_path):
			continue
		with open(out_path , "r") as out_file:
			out_str = out_file.read()
		out_str = out_str.replace("^H","")
		regex_str =r"test\saccuracy\s(\d\.\d+).*loss\s(\d\.\d+).*precision\s.(\d\.\d+).\s(\d\.\d+).\srecall\s.(\d\.\d+).\s(\d\.\d+)." 
		regex = re.compile(regex_str)
		#ret = re.findall(regex, acc_str)
		match = re.search(regex,out_str)
		
		test_acc = match.group(1)
		test_loss = match.group(2)
		test_pre_0 = match.group(3)
		test_pre_1 = match.group(4)
		test_rec_0 = match.group(5)
		test_rec_1 = match.group(6)

		configPath = os.path.join(adir,'netConfig.json')
		if not os.path.exists(configPath):
			continue
		with open(configPath,"r") as f:
			conf = json.loads(f.read())
		csvFile.writerow([os.path.basename(adir), epoch, accuracy,loss_val, precision_0, precision_1, recall_0, recall_1,\
			conf["layer1_stride"], conf["layer2_stride_width"], conf["filter1_height"], conf["filter1_width"], conf["layer2_stride_height"],\
			conf["num_features"], conf["batch_size"], conf["filter2_width"], conf["window_size"], conf["filter1_depth"], conf["drop_rate"],\
			conf["num_epochs"], conf["filter2_depth"], conf["num_classes"], conf["filter2_height"],test_acc,test_loss,test_pre_0,test_pre_1,test_rec_0,test_rec_1 ])
	outFile.close()

def analyzeRunsDataLstm(prefix, path, csvpath):
	dirs = glob.glob(os.path.join(path,prefix)+"*")
	if not dirs:
		raise ValueError("Input path and prefix didn't match any directory. Check input params")
	print "Number of directories found {}".format(len(dirs))
	outFile = open(csvpath,"wb")
	csvFile = csv.writer(outFile, delimiter='\t')
	csvFile.writerow(["run","num_epoch", "best_epoch","train_loss", "train_f1", "val_loss", "val_f1","val_f1_anomaly",\
		"learning_rate","beta_1", "beta_2", "weight_decay", "lstm_size", "drop_w", "drop_u", "batch_size"])

	for adir in dirs:
		output_path = os.path.join(adir, "output.txt")
		if not os.path.exists(output_path):
			continue
		with open(output_path, "r") as op_file:
			op_str = op_file.read()
		#remove unncessary backspaces from the file due to keras
		op_str = op_str.replace("","")
		regex_str = r"val_f1_anomaly:\s.(\S*)\)\W+val_weighted_loss:\s.\S*\W+\s.*step\s-\sloss:\s(\S*)\s-\sf1:\s(\S*)\s-\sval_loss:\s(\S*)\s-\sval_f1:\s(\S*)"
		regex = re.compile(regex_str)
		ret = re.findall(regex, op_str)


		max_val = 0
		max_idx = 0
		#find the max validation f1 
		for i, column in enumerate(ret):
			val_f1 = float(column[4])
			if val_f1 > max_val:
				max_val = val_f1
				max_idx = i

		configPath = os.path.join(adir,'netConfig.json')
		if not os.path.exists(configPath):
			continue
		with open(configPath,"r") as f:
			conf = json.loads(f.read())
		csvFile.writerow([os.path.basename(adir), conf['epoch'], max_idx+1, ret[max_idx][1], ret[max_idx][2], ret[max_idx][3], ret[max_idx][4], ret[max_idx][0], \
		conf['learning_rate'], conf['beta_1'], conf['beta_2'], conf['weight_decay'],conf['lstm_size'], conf['drop_w'], conf['drop_u'], conf['batch_size']])
	outFile.close()

def analyzeRunsDataAutoEncoder(prefix, path, csvpath):
	dirs = glob.glob(os.path.join(path,prefix)+"*")
	if not dirs:
		raise ValueError("Input path and prefix didn't match any directory. Check input params")
	print "Number of directories found {}".format(len(dirs))
	outFile = open(csvpath,"wb")
	csvFile = csv.writer(outFile, delimiter='\t')
	csvFile.writerow(["run", "window_size","overlap","best_epoch", "val_loss","threshold", "precision","recall","best_f1","roc",\
"input_size","layer_1","layer_2","layer_3", "n_estimators", "min_sample_leaf","max_depth"])
	for adir in dirs:
		print adir
		output_path = os.path.join(adir, "output.txt")
		if not os.path.exists(output_path):
			continue
		with open(output_path, "r") as op_file:
			op_str = op_file.read()
		
		regex_f1 = r"Threshold:\s(\d\.\d+e{0,1}-{0,1}\d{0,}),\sPrecision:\s(\d\.\d+e{0,1}-{0,1}\d{0,}),\sRecall:\s(\d\.\d+e{0,1}-{0,1}\d{0,}),\sF1:\s(\d\.\d+e{0,1}-{0,1}\d{0,})"
		regex = re.compile(regex_f1)
		ret_f1 = re.findall(regex, op_str)
		
		
		regex_loss = r"val_loss:\s(\d\.\d+e{0,1}-{0,1}\d{0,})"
		regex = re.compile(regex_loss)
		ret_loss = re.findall(regex, op_str)
		
		regex_roc = r"auc\S:\s\S+\s(\d\.\d+)"
		regex = re.compile(regex_roc)
		ret_roc = re.findall(regex, op_str)	

		max_f1 = 0.0 
		idx_f1 = 0.0
		
		min_loss = 10000.0 
		idx_loss = 0.0
		
		#find largest F1 score
		for i, column in enumerate(ret_f1):
		    f1 = float(column[3])
		    if f1 > max_f1:
		        max_f1 = f1
		        idx_f1 = i	
		
		#find smallest val score
		for i, column in enumerate(ret_loss):
		    loss = float(column)
		    if loss < min_loss:
		        min_loss = loss
		        idx_loss = i	
		
		#find roc
		for i, column in enumerate(ret_roc):
			roc = float(column)	
			print "roc"+str(roc)
		
		configPath_AE = os.path.join(adir,'netConfig_AE.json')
		if not os.path.exists(configPath_AE):
			continue
		with open(configPath_AE,"r") as f:
			conf_AE = json.loads(f.read())

		configPath_RF = os.path.join(adir,'netConfig_RF.json')
		if not os.path.exists(configPath_RF):
			continue
		with open(configPath_RF,"r") as f:
			conf_RF = json.loads(f.read())

		
		regex_win = r"Train file read:\s(\S+)"
		regex = re.compile(regex_win)
		ret_win = re.findall(regex, op_str)
		print ret_win[0]

			
		regex_adir = r"w(\d+)o(\d+)"
		regex = re.compile(regex_adir)
		ret_wo = re.findall(regex, ret_win[0])
		print "hello"
		print ret_wo
		window = ret_wo[0][0]
		overlap = ret_wo[0][1]

		csvFile.writerow([os.path.basename(adir), window, overlap, idx_loss+1, min_loss, ret_f1[idx_f1][0], ret_f1[idx_f1][1],ret_f1[idx_f1][2],\
max_f1,roc,\
conf_AE["input_size"],conf_AE["layer_1"],conf_AE["layer_2"],conf_AE["layer_3"],conf_RF["n_estimators"],conf_RF["min_samples_leaf"],conf_RF["max_depth"]])

	outFile.close()

def analyzeRunsDataSVM(prefix, path, csvpath):
	dirs = glob.glob(os.path.join(path,prefix)+"*")
	if not dirs:
		raise ValueError("Input path and prefix didn't match any directory. Check input params")
	print "Number of directories found {}".format(len(dirs))
	outFile = open(csvpath,"wb")
	csvFile = csv.writer(outFile, delimiter='\t')
	csvFile.writerow(["run", "threshold", "precision","recall","best_f1","roc",\
"degree","coef0","c"])
	for adir in dirs:
		print adir
		output_path = os.path.join(adir, "output.txt")
		if not os.path.exists(output_path):
			continue
		with open(output_path, "r") as op_file:
			op_str = op_file.read()
		
		regex_f1 = r"Threshold:\s(\d\.\d+e{0,1}-{0,1}\d{0,}),\sPrecision:\s(\d\.\d+e{0,1}-{0,1}\d{0,}),\sRecall:\s(\d\.\d+e{0,1}-{0,1}\d{0,}),\sF1:\s(\d\.\d+e{0,1}-{0,1}\d{0,})"
		regex = re.compile(regex_f1)
		ret_f1 = re.findall(regex, op_str)
		
		
		
		regex_roc = r"auc\S:\s\S+\s(\d\.\d+)"
		regex = re.compile(regex_roc)
		ret_roc = re.findall(regex, op_str)	

		max_f1 = 0.0 
		idx_f1 = 0.0
		
		#find largest F1 score
		for i, column in enumerate(ret_f1):
		    f1 = float(column[3])
		    if f1 > max_f1:
		        max_f1 = f1
		        idx_f1 = i	
		
		
		#find roc
		for i, column in enumerate(ret_roc):
			roc = float(column)	
			print "roc"+str(roc)
		
		configPath_SVM = os.path.join(adir,'netConfig_SVM.json')
		if not os.path.exists(configPath_SVM):
			continue
		with open(configPath_SVM,"r") as f:
			conf_SVM = json.loads(f.read())

		csvFile.writerow([os.path.basename(adir), ret_f1[idx_f1][0], ret_f1[idx_f1][1],ret_f1[idx_f1][2],\
max_f1,roc,\
conf_SVM["degree"],conf_SVM["coef0"],conf_SVM["c"]])

	outFile.close()

def analyzeRunsDataRF(prefix, path, csvpath):
	dirs = glob.glob(os.path.join(path,prefix)+"*")
	if not dirs:
		raise ValueError("Input path and prefix didn't match any directory. Check input params")
	print "Number of directories found {}".format(len(dirs))
	outFile = open(csvpath,"wb")
	csvFile = csv.writer(outFile, delimiter='\t')
	csvFile.writerow(["run", "threshold", "precision","recall","best_f1","roc",\
"max_depth","n_estimators","min_samples_leaf"])
	for adir in dirs:
		print adir
		output_path = os.path.join(adir, "output.txt")
		if not os.path.exists(output_path):
			continue
		with open(output_path, "r") as op_file:
			op_str = op_file.read()
		
		regex_f1 = r"Threshold:\s(\d\.\d+e{0,1}-{0,1}\d{0,}),\sPrecision:\s(\d\.\d+e{0,1}-{0,1}\d{0,}),\sRecall:\s(\d\.\d+e{0,1}-{0,1}\d{0,}),\sF1:\s(\d\.\d+e{0,1}-{0,1}\d{0,})"
		regex = re.compile(regex_f1)
		ret_f1 = re.findall(regex, op_str)
		
		
		
		regex_roc = r"auc\S:\s\S+\s(\d\.\d+)"
		regex = re.compile(regex_roc)
		ret_roc = re.findall(regex, op_str)	

		max_f1 = 0.0 
		idx_f1 = 0.0
		
		#find largest F1 score
		for i, column in enumerate(ret_f1):
		    f1 = float(column[3])
		    if f1 > max_f1:
		        max_f1 = f1
		        idx_f1 = i	
		
		
		#find roc
		for i, column in enumerate(ret_roc):
			roc = float(column)	
			print "roc"+str(roc)
		
		configPath_RF = os.path.join(adir,'netConfig_RF.json')
		if not os.path.exists(configPath_RF):
			continue
		with open(configPath_RF,"r") as f:
			conf_RF = json.loads(f.read())

		csvFile.writerow([os.path.basename(adir), ret_f1[idx_f1][0], ret_f1[idx_f1][1],ret_f1[idx_f1][2],\
max_f1,roc,\
conf_RF["max_depth"],conf_RF["n_estimators"],conf_RF["min_samples_leaf"]])

	outFile.close()
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--prefix", help="Provide the name prefix of the runs directories over which\
		hyperparameters analysis has to be done.", required=True)
	parser.add_argument("--path", default = os.path.join(script_dir,"../runs/"), help="Path to where all the \
		individual run directories are present. Defaults to ../runs/")
	parser.add_argument("--csv", help = "Provide the path to the output CSV file that would be created", required = True)
	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument("--lstm", action ="store_true", help = "If analysis is to be done over lstm runs. One of --convnet or --lstm is required")
	group.add_argument("--convnet", action="store_true", help = "if analysis is to be done over convnet runs. One of --convnet or --lstm is required")

	group.add_argument("--autoencoder", action="store_true", help = "if analysis is to be done over autoencoder runs.")

	group.add_argument("--svm", action="store_true", help = "if analysis is to be done over svm runs.")

	group.add_argument("--rf", action="store_true", help = "if analysis is to be done over random forest runs.")

	args = parser.parse_args()

	if args.convnet:
		analyzeRunsDataConvnet(prefix = args.prefix, path = args.path,csvpath = args.csv)
	elif args.lstm:
		analyzeRunsDataLstm(prefix = args.prefix, path = args.path, csvpath = args.csv)
	elif args.autoencoder:
		analyzeRunsDataAutoEncoder(prefix = args.prefix, path = args.path, csvpath = args.csv)
	elif args.svm:
		analyzeRunsDataSVM(prefix = args.prefix, path = args.path, csvpath = args.csv)
	elif args.rf:
		analyzeRunsDataRF(prefix = args.prefix, path = args.path, csvpath = args.csv)

