import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import random
from sklearn.model_selection import train_test_split, KFold

from fileutil import *

PATH = "./pytorch-model.pth"

os.environ["KMP_DUPLICATE_LIB_OK"]="True"

#body_labels = ['idle', 'both-arms-up', 'arm-up-right', "arm-up-left", "show-right", "show-left", "show-up-right", "show-up-left", "clap", "cheer", 
#"complain", "both-arms-right", "both-arms-left", "t-pose", "fists-together", "arm-bow-right", "arm-bow-left", "cross-arms", "time-out-low", "time-out-high"]

body_labels = ['clap', 'spin']

def label_to_number(label):
	return body_labels.index(label)

dataset = "pose-4"

input_size = 16

output_size = len(body_labels)

k_fold_splits = 4

validation_iter = 1

batch_size = 10

learning_rate = 0.01

class TrainEntry:
	def __init__(self, label, tensor):
		self.label = label
		self.tensor = tensor

class Model(nn.Module):
	def __init__(self, input_size, num_layers, hidden_size, output_size):
		super(Model, self).__init__()
		self.i2h = nn.Linear(input_size, hidden_size)

		h2h_layers = [] 
		for i in range(num_layers - 1):
			h2h = nn.Linear(hidden_size, hidden_size)
			h2h_layers.append(h2h) 
		self.h2h_layers = h2h_layers

		self.h2o = nn.Linear(hidden_size, output_size)
		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, x):
		x = self.i2h(x)

		for h2h in self.h2h_layers:
			x = F.relu(h2h(x))

		x = self.h2o(x)
		x = self.softmax(x)
		return x

def get_input():

	body_entries = get_all_body_entries(dataset)

	train_entries = []
	for body_entry in body_entries:

		inputs = np.array([body_entry.angles])
		inputs = (inputs * 2) - 1
		labels = np.array([label_to_number(body_entry.label)])

		label = torch.from_numpy(labels)
		tensor = torch.from_numpy(inputs)
		tensor = tensor.float()

		train_entries.append(TrainEntry(label, tensor))
	return np.array(train_entries)

def batch_entries(batch_size, entries):

	batched_entries = []
	for i in range(0, len(entries), batch_size):

		inputs_batch = []
		label_batch = []
		for j in range(i, i + batch_size):
			if j >= len(entries):
				break

			inputs = entries[j].tensor[0].numpy()
			inputs_batch.append(inputs)
			label = entries[j].label[0]
			label_batch.append(label)

		tensor = torch.tensor(inputs_batch)
		label = torch.tensor(label_batch)
		batched_entries.append(TrainEntry(label, tensor))
	return np.array(batched_entries)

def train_model(num_epochs, num_layers, hidden_size, train_entries, print_loss = True):

	model = Model(input_size, num_layers, hidden_size, output_size)
	model = model.float()

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr = learning_rate)	

	for epoch in range(num_epochs):

		random.shuffle(train_entries)
		batched_entries = batch_entries(batch_size, train_entries)

		running_loss = 0
		for train_entry in batched_entries:
			optimizer.zero_grad()
			outputs = model(train_entry.tensor)
			loss = criterion(outputs, train_entry.label)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
		if print_loss:
			print("Epoch %i: %f loss" % (epoch, running_loss))
	return model

def validate_classes(num_epochs, num_layers, hidden_size):

	entries = get_input()

	accuracys = []
	label_correct_acc = np.zeros(output_size)
	label_total_acc = np.zeros(output_size)
	for i in range (validation_iter):

		print("validation - num_epochs: %d, num_layers: %d, hidden_size: %d - iteration %d/%d"  % (num_epochs, num_layers, hidden_size, (i + 1), validation_iter))
		kf = KFold(n_splits = k_fold_splits, shuffle = True)
		for train_index, test_index in kf.split(entries):
	
			train_entries = entries[train_index]
			test_entries = entries[test_index]

			model = train_model(num_epochs, num_layers, hidden_size, train_entries)
			accuracy, label_correct, label_total = get_accuracy_with_labels(model, test_entries)
			accuracys.append(accuracy)
			label_correct_acc += label_correct
			label_total_acc += label_total

	mean = np.mean(accuracys)
	std = np.std(accuracys)

	result = "---------- \nnum_epochs: %d, num_layers: %d, hidden_size: %d \nmean: %.1f, std: %.1f"  % (num_epochs, num_layers, hidden_size, mean, std)
	for i in range(output_size):
		print('Overall accuracy of %s (%d): %d %%' % (body_labels[i], label_total_acc[i], 100 * label_correct_acc[i] / label_total_acc[i]))

def validate_model(num_epochs, num_layers, hidden_size):

	entries = get_input()

	accuracys = []
	for i in range (validation_iter):

		print("validation - num_epochs: %d, num_layers: %d, hidden_size: %d - iteration %d/%d"  % (num_epochs, num_layers, hidden_size, (i + 1), validation_iter))
		kf = KFold(n_splits = k_fold_splits, shuffle = True)
		for train_index, test_index in kf.split(entries):
	
			train_entries = entries[train_index]
			test_entries = entries[test_index]

			model = train_model(num_epochs, num_layers, hidden_size, train_entries)
			accuracy = get_accuracy(model, test_entries)
			accuracys.append(accuracy)




	mean = np.mean(accuracys)
	std = np.std(accuracys)

	result = "---------- \nnum_epochs: %d, num_layers: %d, hidden_size: %d \nmean: %.1f, std: %.1f"  % (num_epochs, num_layers, hidden_size, mean, std)

	return result

def get_accuracy(model, test_entries):

	correct = 0
	with torch.no_grad():
		for test_entry in test_entries:

			label = test_entry.label.item()
			outputs = model.forward(test_entry.tensor)
			_, predicted = torch.max(outputs, 1)
			if (predicted == label):
				correct += 1

	accuracy = 100 * correct / len(test_entries)
	return accuracy

def get_accuracy_with_labels(model, test_entries):

	print(len(test_entries))
	correct = 0
	label_correct = np.zeros(output_size)
	label_total = np.zeros(output_size)

	with torch.no_grad():
		for test_entry in test_entries:
			
			label = test_entry.label.item()
			outputs = model.forward(test_entry.tensor)
			_, predicted = torch.max(outputs, 1)
			if (predicted == label):
				correct += 1
			if (predicted == label):
				label_correct[label] += 1
			label_total[label] += 1

	accuracy = 100 * correct / len(test_entries)
	return accuracy, label_correct, label_total

def hyperopt():

	nums_epochs = [100]#[10, 25, 50 , 100]
	hidden_sizes = [40]#[5, 10, 20, 30]
	nums_layers = [3]#[1, 2, 3]

	results = []

	for num_epochs in nums_epochs:
		for hidden_size in hidden_sizes:
			for num_layers in nums_layers:

				result = validate_model(num_epochs, num_layers, hidden_size)
				results.append(result)

	print("---------- RESULTS ----------")
	for result in results:
		print(result)

#-------- IMPORT IMAGES --------#

print("---------- START ----------")

#hyperopt()

validate_classes(1, 2, 1)
#print(result)








