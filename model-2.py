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

PATH = './pytorch-model.pth'

os.environ['KMP_DUPLICATE_LIB_OK']='True'

body_labels = ['idle', 'both-arms-up', 'arm-up-right', "arm-up-left", "show-right", "show-left", "show-up-right", "show-up-left", "clap", "cheer", 
"complain", "both-arms-right", "both-arms-left", "t-pose", "fists-together", "arm-bow-right", "arm-bow-left", "cross-arms", "time-out-low", "time-out-high"]

num_of_labels = len(body_labels)

input_size = 16

hidden_size = 20

k = 4

def label_to_number(label):
	if(label == 'idle'):
		return 0
	if(label == 'both-arms-up'):
		return 1
	if(label == 'arm-up-right'):
		return 2
	if(label == 'arm-up-left'):
		return 3
	if(label == 'show-right'):
		return 4
	if(label == 'show-left'):
		return 5
	if(label == 'show-up-right'):
		return 6
	if(label == 'show-up-left'):
		return 7
	if(label == 'clap'):
		return 8
	if(label == 'cheer'):
		return 9
	if(label == 'complain'):
		return 10
	if(label == 'both-arms-right'):
		return 11
	if(label == 'both-arms-left'):
		return 12
	if(label == 't-pose'):
		return 13
	if(label == 'fists-together'):
		return 14
	if(label == 'arm-bow-right'):
		return 15
	if(label == 'arm-bow-left'):
		return 16
	if(label == 'cross-arms'):
		return 17
	if(label == 'time-out-low'):
		return 18
	if(label == 'time-out-high'):
		return 19

def get_batch(all_entries, n):

	all_inputs = []
	all_labels = []
	for i in range (0, len(all_entries), n):
		entries = all_entries[i:(i+n)]
		inputs = np.array(list(map(lambda e: e.angles, entries)))
		inputs = (inputs * 2) - 1
		labels = np.array(list(map(lambda e: label_to_number(e.label), entries)))

		t_inputs = torch.from_numpy(inputs)
		t_labels = torch.from_numpy(labels)

		all_inputs.append(t_inputs)
		all_labels.append(t_labels)

	return all_inputs, all_labels

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(input_size, hidden_size)
		self.fc2 = nn.Linear(hidden_size, hidden_size)
		self.fc3 = nn.Linear(hidden_size, hidden_size)
		self.fc4 = nn.Linear(hidden_size, num_of_labels)
		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = self.fc4(x)
		x = self.softmax(x)
		return x

def train_model(entries):

	net = Net()
	net = net.double()

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(net.parameters(), lr=0.001)	

	for epoch in range(30):
		# shuffle entries and create batches
		random.shuffle(entries)
		all_inputs, all_labels = get_batch(entries, 1)
		running_loss = 0
		for i in range (len(all_inputs)):

			inputs = all_inputs[i]
			labels = all_labels[i]

			optimizer.zero_grad()
			outputs = net(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()

		print("Epoch %i: %f loss" % (epoch, running_loss))
	return net

def validate_model():

	entries = np.array(get_all_body_entries("pose-2"))
	kf = KFold(n_splits = k, shuffle = True)
	percentage_acc = 0
	label_correct_acc = np.zeros(num_of_labels)
	label_total_acc = np.zeros(num_of_labels)
	for train_index, test_index in kf.split(entries):

		train_entries = entries[train_index]
		test_entries = entries[test_index]


		net = train_model(train_entries)
		percentage, label_correct, label_total = print_results(net, test_entries)
		percentage_acc += percentage
		label_correct_acc += label_correct
		label_total_acc += label_total

	for i in range(num_of_labels):
		print('Overall accuracy of %s (%d): %d %%' % (body_labels[i], label_total_acc[i], 100 * label_correct_acc[i] / label_total_acc[i]))
	print('Overall accuracy of the network: %.1f %%' % (percentage_acc / k))

def print_results(net, entries):

	all_inputs, all_labels = get_batch(entries, 1)

	correct = 0
	total = len(entries)
	label_correct = np.zeros(num_of_labels)
	label_total = np.zeros(num_of_labels)
	with torch.no_grad():
		for i in range(len(all_inputs)):
			inputs = all_inputs[i]
			labels = all_labels[i]
			label = labels.item()
			outputs = net.forward(inputs)
			_, predicted = torch.max(outputs, 1)
			if (predicted == label):
				correct += 1
			else:
				print('Label: ' + body_labels[label] + ", predicted: " + body_labels[predicted])
			if (predicted == label):
				label_correct[label] += 1
			label_total[label] += 1

	percentage = 100 * correct / total
	#print('Accuracy of the network on the ' + str(total) + ' test images: %.1f %%' % percentage)
	#for i in range(8):
		#print('Accuracy of %s (%d): %d %%' % (body_labels[i], label_total[i], 100 * label_correct[i] / label_total[i]))
	return percentage, label_correct, label_total


#-------- IMPORT IMAGES --------#

print("---------- START ----------")

validate_model()






