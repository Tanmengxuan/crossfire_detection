import glob
import csv
import re
import numpy as np
from collections import Counter
import pandas as pd
#import keras as keras
from sklearn import preprocessing


class inputter:
	
	def __init__(self,filepathGlob):
		files = glob.glob(filepathGlob)
		#pdb.set_trace()
		self.trainFile = ''
		self.validationFile = ''
		self.testFile = ''
		self.windowSize = None
		self.overlap = None
		for file in files:
			if re.search(r'test\.csv',file):
				self.testFile = file
				print "Test file read: "+file
				self.extractWindowOverlap(file)
			elif re.search(r'train\.csv',file):
				self.trainFile = file
				print "Train file read: " + file
				self.extractWindowOverlap(file)
			elif re.search(r'validation\.csv',file):
				self.validationFile = file
				print "Validation file read: "+file
				self.extractWindowOverlap(file)
		#assert(self.trainFile), "Training file not found, check Glob passed"
		#assert(self.validationFile)	, "Validation file not found, check Glob passed"
		#assert(self.testFile), "Test File not found, check Glob passed"

	def extractWindowOverlap(self,file):
		m = re.search(r'_w(\d+)o(\d+)_',file)
		assert(m), "Couldn't find window size and overlap size in file " + file
		if self.windowSize is None:
			self.windowSize = int(m.group(1))
			self.overlap = int(m.group(2))
		else:
			assert (int(m.group(1)) == self.windowSize and int(m.group(2)) == self.overlap),"Window or overlap size not same for all files"
	
	def determineTrainWindows(self):
		if not self.trainFile:
			raise ValueError("No training file has been read, please specify")
		with open(self.trainFile,"r") as trainFile:
			num_lines = sum(1 for line in trainFile)
		return num_lines/self.windowSize
	

	def trainingBatches(self, batchSize):
		if not self.trainFile:
			raise ValueError("No training file has been read, please specify")
		noloop = 0
		return self.generateBatches(self.trainFile, batchSize, noloop )

	def validationBatches(self,batchSize):
		if not self.validationFile:
			raise ValueError("No validation file has been read, please specify")
		loop= 1
		return self.generateBatches(self.validationFile, batchSize, loop)

	def getValidationData(self, mergeAttackLabelsWithAtleast=False):
		if not self.validationFile:
			raise ValueError("No Validation file has been read, please specify")
		return inputWindowedData(self.validationFile,self.windowSize, mergeAttackLabelsWithAtleast, forTest=False)

	def getTestData(self, mergeAttackLabelsWithAtleast=False):
		if not self.testFile:
			raise ValueError("No test file has been read, please specify")
		return inputWindowedData(self.testFile, self.windowSize, mergeAttackLabelsWithAtleast, forTest=False)

	def getTrainData(self,mergeAttackLabelsWithAtleast=False):
		if not self.trainFile:
			raise ValueError("No train file has been read, please specify")
		return inputWindowedData(self.trainFile, self.windowSize, mergeAttackLabelsWithAtleast, forTest=False)
    
def inputWindowedData(filePath,windowSize,mergeAttackLabelsWithAtleast=False, forTest=False):
	with open(filePath,"r") as file:
        # initiate the list of data and label for the given dataset
		data=[]
		labels=[]
        # read the dataset in
		pd_data = pd.read_csv(file,sep="\t", index_col=None, header=None, dtype='float64')
        # print some signal to the console window
		print ('read file in')
        # initiate the while loop index
		i=0
        # while the loop index is smaller than the size on the dataset, buffer 2 windowSize to prevent index out of bound
		while i < pd_data.shape[0]-2*windowSize-2:
            # store 1 windowSize of data and label near i
            # use iloc to get the specific location
            # use .values to get the value in floating point type
			previous_window = pd_data.iloc[i:i+windowSize, 1:].values
			previous_label = pd_data.iloc[i:i+windowSize,0].values
            # initialize the one valid windowSize of data and label
			valid_window=[]
			valid_label=[]
            # initialize the flag of finding an end point of a warm up period
			found_end_point = False
            # if forTest == True (meaning this data table is for Test purpose
            # of if the next chunk of windowSize contains all zero labels
            # then append the near chunk of windowSize to the final data and labels lists
			if forTest or sum(pd_data.iloc[i+windowSize:i+2*windowSize,0]) == 0:
				valid_window = previous_window
				valid_label = previous_label
                # use np.array to turn a list to an array, for data handling later on
				data.append(np.array(valid_window))
                # use np.reshape to turn 1-dimensional array to 2-dimensional array
				valid_label= np.reshape(valid_label, [valid_label.shape[0],1])
                # turn each value of 0 or 1 into one_hot vector of [1,0] or [0,1]
				valid_label = np.append(1-valid_label, valid_label, axis=1)
				labels.append(valid_label)
                # increase the loop counter
				i = i+windowSize
            # if not forTest, and the next chunk of windowSize contains at least one label '1'
			else:
                # search within the next chunk of windowSize
				for j in range (i+windowSize, i+ 2*windowSize):
                    # try to find the labels pattern "10", which signals the end of a warm-up period
					if ((pd_data.iloc[j,0]==1) and (pd_data.iloc[j+1,0]==0)):
                        # get the chunk with the end of the warm-up period to be the last data
                        # use iloc to get the specific location
                        # use .values to get the value in floating point type
						valid_window = pd_data.iloc[j-windowSize+1:j+1, 1:].values       
						valid_label = pd_data.iloc[j-windowSize+1:j+1,0].values
                        # use np.array to tunr a list into an array
						data.append(np.array(valid_window))
                        # use np.shape to turn a 1-dimensional array to 2-dimensional array
						valid_label= np.reshape(valid_label, [valid_label.shape[0],1])
                        # turn each value to 0 or 1 into one_hot vector of [1,0] or [0,1]
						valid_label = np.append(1-valid_label, valid_label, axis=1)
						labels.append(valid_label)
                        # flag the end point
						found_end_point=True
                        # go to the end of the for loop
						break
                # if found the end point, append the new chunk data and then update the while loop index
				if found_end_point:
					i = j
                # if not found end point, increase the while loop index to continue searching for the end point
				else:
					i = i+1
		print ('finish windowing')    
    # use np.array to turn data list into an array
	datanp = np.array(data)
	del data
	labelsnp = (np.array(labels))
    # print out the summary of how many zeros and ones labels in the dataset
	print np.sum(np.sum(labelsnp, axis=0), axis=0)    
	shapeData = datanp.shape
	datanp = datanp.reshape(shapeData[0],shapeData[1],shapeData[2],1)
	return [datanp,labelsnp]
