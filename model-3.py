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

classes = ["spin", "clap", "time-out", "dance", "idle", "fold"]

def label_to_number(label):
	return classes.index(label)

dataset = "pose-4"

input_size = 16

output_size = len(classes)

k_fold_splits = 4

validation_iter = 10

sequence_length = 10

batch_size = 10

learning_rate = 0.005

class TrainEntry:
	def __init__(self, label, tensor):
		self.label = label
		self.tensor = tensor

class RNN(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers):
		super().__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers

		self.RNN = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True) #inputs and outputs are  (batch, seq, feature)
		self.r2h = nn.Linear(hidden_size, hidden_size)
		self.h2o = nn.Linear(hidden_size, output_size)
		self.softmax = nn.LogSoftmax(dim=1)
		
	def forward(self, x):
		hidden_state = torch.zeros([self.num_layers, x.shape[0], self.hidden_size])
		x, h = self.RNN(x, hidden_state)
		x = F.relu(self.r2h(x[:,-1,:])) # get last output
		x = self.h2o(x)
		x = self.softmax(x)
		return x


def get_input(log = False):

	train_entries = []

	for label in classes:
		entries = get_body_entries(dataset, label)

		if log: 
			print("Found %d frames for label %s" % (len(entries), label))

		sequences = []
		for i in range(0, len(entries), sequence_length):
			frames = []
			for j in range(i, i + sequence_length):
				frames.append((entries[j].angles * 2) - 1)


			tensor = torch.tensor([frames])
			tensor = tensor.float()
			train_entries.append(TrainEntry(torch.tensor([label_to_number(label)]), tensor))

	if log: 
		print("Created %d clips with length %d" % (len(train_entries), sequence_length))

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


def train_model(num_epochs, num_layers, hidden_size, train_entries, log = False):

	model = RNN(input_size, hidden_size, num_layers)
	model = model.float()

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr = learning_rate)	

	for epoch in range(num_epochs):

		random.shuffle(train_entries)
		batched_entries = batch_entries(batch_size, train_entries)

		running_loss = 0
		for train_entry in batched_entries:
			optimizer.zero_grad()
			hState = torch.zeros([num_layers, train_entry.tensor.shape[0], hidden_size])
			outputs = model(train_entry.tensor)
			loss = criterion(outputs, train_entry.label)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
		if log:
			print("Epoch %i: %f loss" % (epoch, running_loss))
	return model


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

	return accuracys


def hyperopt():



	num_layers = 1

	nums_epochs = [25,50,100,200]
	hidden_sizes = [10, 40]
	nums_layers = [1]

	results = []

	for num_epochs in nums_epochs:
		for hidden_size in hidden_sizes:
			for num_layers in nums_layers:
				iteration = "num_epochs: %d, num_layers: %d, hidden_size: %d"  % (num_epochs, num_layers, hidden_size)
				print(iteration)

				accuracys = validate_model(num_epochs, num_layers, hidden_size)

				mean = np.mean(accuracys)
				std = np.std(accuracys)

				results.append(iteration)
				results.append("Mean: %.2f, std: %.2f" % (mean, std))

	print("---------- RESULTS ----------")
	for result in results:
		print(result)



print("---------- START ----------")

hyperopt()