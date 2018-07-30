import json
import os.path
import subprocess
import re
import random
import time
import sys
import os
import numpy as np

lstm_size = [1000, 1200, 1300,1400,1500,1600,1700,1800,1900,2000]
learning_rate = [0.01, 0.001, 0.005]
beta_1 = [0.85, 0.87, 0.9, 0.92, 0.55, 0.63, 0.77, 0.2, 0.3]
beta_2 = [0.01, 0.001, 0.005]
weight_decay = [0.01, 0.05, 0.1, 0.15, 0.2]
drop_w = [0.01, 0.2,  0.65]
drop_u = [0.0, 0.2,  0.65]
batch_size = [512] 
num_epoch = [30]

configsTried = []
i=0
while i < 288:
	lstm_size_l = random.choice(lstm_size)
	learning_rate_l = 10**(np.random.uniform(-4.0, 0)) 
	beta_1_l = random.choice(beta_1) 
	beta_2_l = 1 - 10**(np.random.uniform(-3.0 , -1.0)) 
	weight_decay_l = random.choice(weight_decay)
	drop_w_l = random.choice(drop_w)
	drop_u_l = random.choice(drop_u)
	batch_size_l = random.choice(batch_size)
	num_epoch_l = random.choice(num_epoch)
	
	cfg = {
		"input_path" : "/home/cuc/CrossFire-Detect/CrossFire-Detect/models/data/trafficData/servers_1000/30_1000decoys_localmean/1000decoys_0_0_1_noise/normed*",
		"lstm_size" : lstm_size_l,
		"n_inputs" : 64,
		"learning_rate" : learning_rate_l,
		"beta_1" : beta_1_l,
		"beta_2" : beta_2_l,
		"weight_decay" : weight_decay_l,
		"num_features" : 1000,
		"drop_rate" : 0.5, 
		"drop_w" : drop_w_l,
		"drop_u" : drop_u_l,
		"epoch" : num_epoch_l,
		"batch_size" : batch_size_l 
	}
	configstr = json.dumps(cfg)
	if configstr in configsTried:
			continue
	#configsTried.append(configstr)
	with open("curRunConfig_lstm.json","wb") as f:
		f.write(configstr)
	try:
		print "Trying with config {}".format(cfg)
		print i
		i += 1
		start_time = time.time()
		#with open('output.txt', 'w') as f:
			#op = subprocess.Popen("python lstm_classify.py",shell= True, stdout=subprocess.PIPE)
			#op = subprocess.Popen("python lstm_classify.py",shell = True, stdout=subprocess.PIPE)
			#op = subprocess.Popen("python lstm_classify.py", shell=True,stderr=subprocess.STDOUT)
			#a = os.system("python lstm_classify.py")
			#print a
		#for c in iter(lambda: op.stdout.read(1), ''):  # replace '' with b'' for Python 3
		#	sys.stdout.write(c)
		#	f.write(c)
		#op = subprocess.check_output("python lstm_classify.py", shell=True,stderr=subprocess.STDOUT)

		#proc = subprocess.Popen('python lstm_classify.py', shell=True, stdout=subprocess.PIPE )
		#output = proc.communicate()[0]
		#print output  
		output = os.popen('python lstm_classify.py').read()
		
		print("--- %s minutess ---" % ((time.time() - start_time)/60.0))
		m = re.search(r'Results\sstored\sin\s(.*runs.*)',output)
		if m:
			op_path = m.group(1)
			print op_path
			print ("\n")
			with open(os.path.join(op_path,"output.txt"),"w") as outFile:
				outFile.write(output)

			configsTried.append(configstr)
	except:
		pass
