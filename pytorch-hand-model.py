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

hand_labels = ['one', 'two', 'three', 'four', 'five']

def label_to_number(label):
	if(label == 'one'):
		return 0
	if(label == 'two'):
		return 1
	if(label == 'three'):
		return 2
	if(label == 'four'):
		return 3
	if(label == 'five'):
		return 4

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
		self.fc1 = nn.Linear(15, 15)
		self.fc2 = nn.Linear(15, 15)
		self.fc3 = nn.Linear(15, 5)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

def train_model(entries):

	net = Net()
	net = net.double()

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(net.parameters(), lr=0.001)	

	for epoch in range(50):
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

		#print("Epoch %i: %f loss" % (epoch, running_loss))
	return net

def validate_model():

	entries = np.array(get_all_hand_entries())
	k = 5
	kf = KFold(n_splits = k, shuffle = True)
	percentage_acc = 0
	label_correct_acc = np.array([0, 0, 0, 0, 0])
	label_total_acc = np.array([0, 0, 0, 0, 0])
	for train_index, test_index in kf.split(entries):

		train_entries = entries[train_index]
		test_entries = entries[test_index]


		net = train_model(train_entries)
		percentage, label_correct, label_total = print_results(net, test_entries)
		percentage_acc += percentage
		label_correct_acc += label_correct
		label_total_acc += label_total

	for i in range(5):
		print('Overall accuracy of %s (%d): %d %%' % (hand_labels[i], label_total_acc[i], 100 * label_correct_acc[i] / label_total_acc[i]))
	print('Overall accuracy of the network: %.1f %%' % (percentage_acc / k))

def print_results(net, entries):

	inputs_batch, labels_batch = get_batch(entries, 1)

	correct = 0
	total = len(entries)
	label_correct = np.array([0, 0, 0, 0, 0])
	label_total = np.array([0, 0, 0, 0, 0])
	with torch.no_grad():
		for i in range(len(inputs_batch)):
			inputs = inputs_batch[i]
			correct_labels = labels_batch[i]
			label = correct_labels.item()
			outputs = net.forward(inputs)
			_, predicted = torch.max(outputs, 1)
			if (predicted == label):
				correct += 1
			if (predicted == label):
				label_correct[label] += 1
			label_total[label] += 1

	percentage = 100 * correct / total
	print('Accuracy of the network on the ' + str(total) + ' test images: %.1f %%' % percentage)
	#for i in range(5):
	#	print('Accuracy of %s (%d): %d %%' % (hand_labels[i], label_total[i], 100 * label_correct[i] / label_total[i]))
	return percentage, label_correct, label_total



#---------- START ----------

print("---------- START ----------")

validate_model()






