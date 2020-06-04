import torch
import torchvision
import torchvision.transforms as transforms

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

body_labels = ["idle", "t-pose", "dab", "clap", "show-left", "show-right", "arm-up-left", "arm-up-right"]

def label_to_number(label):
	return body_labels.index(label)

dataset = "pose-1"

input_size = 16

output_size = len(body_labels)

k_fold_splits = 4

validation_iter = 10

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
		x = F.relu(self.i2h(x))

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
		label = torch.tensor(label_batch, dtype=torch.long)
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

def validate_model(num_epochs, num_layers, hidden_size):

	entries = get_input()

	accuracys = []
	for i in range (validation_iter):
		kf = KFold(n_splits = k_fold_splits, shuffle = True)
		for train_index, test_index in kf.split(entries):
			print(test_index)
			train_entries = entries[train_index]
			test_entries = entries[test_index]

			model = train_model(num_epochs, num_layers, hidden_size, train_entries)
			accuracy = get_accuracy(model, test_entries)
			accuracys.append(accuracy)

	return accuracys

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

def hyperopt():

	nums_epochs = [100, 200, 400]#[10, 25, 50, 100, 200]
	hidden_sizes = [20, 30]
	nums_layers = [2]

	results = []


	for num_layers in nums_layers:
		for hidden_size in hidden_sizes:
			for num_epochs in nums_epochs:
				iteration = "num_layers: %d, hidden_size: %d, num_epochs: %d"  % (num_layers, hidden_size, num_epochs)
				print(iteration)

				accuracys = validate_model(num_epochs, num_layers, hidden_size)

				mean = np.mean(accuracys)
				std = np.std(accuracys)

				results.append(iteration)
				results.append("Mean: %.1f, std: %.1f" % (mean, std))

	print("---------- RESULTS ----------")
	for result in results:
		print(result)

#-------- IMPORT IMAGES --------#

print("---------- START ----------")

hyperopt()






