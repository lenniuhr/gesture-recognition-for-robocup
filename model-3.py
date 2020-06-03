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

body_labels = ["spin", "clap", "time-out", "dance", "idle", "fold"]

def label_to_number(label):
	return body_labels.index(label)

dataset = "pose-4"

input_size = 16

output_size = len(body_labels)

k_fold_splits = 4

validation_iter = 1

sequence_length = 10

batch_size = 10

learning_rate = 0.01

num_layers = 1

class TrainEntry:
	def __init__(self, label, tensor):
		self.label = label
		self.tensor = tensor

class RNN(nn.Module):
	def __init__(self, input_size, hidden_size, num_layers):
		super().__init__()
		self.RNN = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True) #inputs and outputs are  (batch, seq, feature)
		self.r2h = nn.Linear(hidden_size, hidden_size)
		self.h2o = nn.Linear(hidden_size, output_size)
		self.softmax = nn.LogSoftmax(dim=1)
		
	def forward(self, x, hidden_state):
		x, h = self.RNN(x, hidden_state)
		x = F.relu(self.r2h(x[:,-1,:])) # gets last output
		x = self.h2o(x)
		x = self.softmax(x)
		return x


def train_model(num_epochs, num_layers, hidden_size, train_entries, print_loss = False):

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
			#print(train_entry.tensor.shape)
			hState = torch.zeros([num_layers, batch_size, hidden_size])
			outputs = model(train_entry.tensor, hState)
			#print(train_entry.label)
			loss = criterion(outputs, train_entry.label)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
		if print_loss:
			print("Epoch %i: %f loss" % (epoch, running_loss))
	return model


def get_input():

	train_entries = []

	for label in body_labels:
		entries = get_body_entries(dataset, label)

		sequences = []
		for i in range(0, len(entries), sequence_length):
			frames = []
			for j in range(i, i + sequence_length):
				#print(entries[j].label + " - " + str(entries[j].frame_nr))
				frames.append((entries[j].angles * 2) - 1)


			tensor = torch.tensor([frames])
			tensor = tensor.float()
			train_entries.append(TrainEntry(torch.tensor([label_to_number(label)]), tensor))

	return np.array(train_entries)


def batch_entries(batch_size, entries):
	batched_entries = []
	for i in range(0, len(entries), batch_size):

		inputs_batch = []
		label_batch = []
		for j in range(i, i + batch_size):
			if j >= len(entries):
				return np.array(batched_entries)

			#print(entries[j].tensor)
			#print(entries[j].label)

			inputs = entries[j].tensor[0].numpy()
			inputs_batch.append(inputs)
			label = entries[j].label[0]
			label_batch.append(label)

		tensor = torch.tensor(inputs_batch)
		label = torch.tensor(label_batch, dtype=torch.long)
		batched_entries.append(TrainEntry(label, tensor))
	return np.array(batched_entries)

def get_accuracy(model, test_entries):

	correct = 0
	with torch.no_grad():
		for test_entry in test_entries:

			label = test_entry.label.item()
			hidden_state = torch.zeros([num_layers, 1, 20])
			outputs = model.forward(test_entry.tensor, hidden_state)
			_, predicted = torch.max(outputs, 1)
			if (predicted == label):
				correct += 1

	accuracy = 100 * correct / len(test_entries)
	return accuracy

def validate_model(num_epochs, num_layers, hidden_size):

	entries = get_input()

	print("Found %d entries" % len(entries))

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

def hyperopt():

	nums_epochs = [50]
	hidden_sizes = [20]#[3, 5, 10, 20]

	results = []

	for num_epochs in nums_epochs:
		for hidden_size in hidden_sizes:
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

#-------- IMPORT IMAGES --------#

print("---------- START ----------")

hyperopt()