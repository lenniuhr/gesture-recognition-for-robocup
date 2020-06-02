import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import os
import sys
import random
from sklearn.model_selection import train_test_split, KFold

from fileutil import *

PATH = './pytorch-model.pth'

os.environ['KMP_DUPLICATE_LIB_OK']='True'

body_labels = ['idle', 'clap']

num_of_labels = len(body_labels)

class TrainEntry:
	def __init__(self, label, tensor):
		self.label = label
		self.tensor = tensor

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

def get_entry_list(all_entries):
	all_inputs = []
	all_labels = []
	for i in range (0, len(all_entries)):
		inputs = all_entries[i].angles
		inputs = (inputs * 2) - 1
		label = label_to_number(all_entries[i].label)
		input_tensor = torch.tensor([[inputs, inputs, inputs]])
		label_tensor = torch.tensor([label])

		all_inputs.append(input_tensor)
		all_labels.append(label_tensor)

	return all_inputs, all_labels

def get_batch(all_entries, n):

	all_inputs = []
	all_labels = []
	for i in range (0, len(all_entries), n):
		entries = all_entries[i:(i+n)]
		
		inputs = np.array(list(map(lambda e: e.angles, entries)))
		inputs = (inputs * 2) - 1
		labels = np.array(list(map(lambda e: label_to_number(e.label), entries)))

		print(list(map(lambda e: e.angles, entries)))
		t_inputs = torch.from_numpy(np.array([inputs, inputs, inputs]))
		t_labels = torch.from_numpy(np.array([labels, labels, labels]))

		all_inputs.append(t_inputs)
		all_labels.append(t_labels)

	return all_inputs, all_labels

class RNN(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(RNN, self).__init__()

		self.hidden_size = hidden_size

		self.i2h = nn.Linear(input_size + hidden_size, hidden_size)

		self.i2o = nn.Linear(input_size + hidden_size, input_size + hidden_size)
		self.i2o2 = nn.Linear(input_size + hidden_size, output_size)
		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, input, hidden):
		combined = torch.cat((input, hidden), 1)
		hidden = self.i2h(combined)
		output = self.i2o(combined)
		output = self.i2o2(output)
		output = self.softmax(output)
		#hidden = output
		return output, hidden

	def initHidden(self):
		return Variable(torch.zeros(1, self.hidden_size))

class RNN2(nn.Module):
	def __init__(self, input_size, hidden_size, output_size):
		super(RNN, self).__init__()

		self.hidden_size = hidden_size

		self.recurrent = nn.Linear(output_size, output_size)
		self.fc1 = nn.Linear(input_size, hidden_size)
		self.fc2 = nn.Linear(hidden_size, hidden_size)
		self.fc3 = nn.Linear(hidden_size, output_size)
		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, input, hidden):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		x = self.softmax(x)
		return output, hidden

	def initHidden(self):
		return Variable(torch.zeros(1, self.hidden_size))

def train_model(entries):

	net = Model(5, 8, 15, 2)
	net = net.float()

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(net.parameters(), lr=0.001)	

	for epoch in range(25):
		# shuffle entries and create batches
		random.shuffle(entries)
		all_inputs, all_labels = get_entry_list(entries)
		running_loss = 0
		for i in range (len(all_inputs)):

			inputs = all_inputs[i]
			labels = all_labels[i]

			optimizer.zero_grad()
			out, hidden = net(inputs.float())
			loss = criterion(out, labels)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()

		#print("Epoch %i: %f loss" % (epoch, running_loss))
	return net

def validate_model():

	entries = np.array(get_all_body_entries("pose-1"))
	k = 5
	kf = KFold(n_splits = k, shuffle = True)
	percentage_acc = 0
	label_correct_acc = np.array([0, 0, 0, 0, 0, 0, 0, 0])
	label_total_acc = np.array([0, 0, 0, 0, 0, 0, 0, 0])
	for train_index, test_index in kf.split(entries):

		train_entries = entries[train_index]
		test_entries = entries[test_index]


		net = train_model(train_entries)
		percentage, label_correct, label_total = print_results(net, test_entries)
		percentage_acc += percentage
		label_correct_acc += label_correct
		label_total_acc += label_total

	for i in range(8):
		print('Overall accuracy of %s (%d): %d %%' % (body_labels[i], label_total_acc[i], 100 * label_correct_acc[i] / label_total_acc[i]))
	print('Overall accuracy of the network: %.1f %%' % (percentage_acc / k))

def print_results(net, entries):

	all_inputs, all_labels = get_batch(entries, 1)

	correct = 0
	total = len(entries)
	label_correct = np.array([0, 0, 0, 0, 0, 0, 0, 0])
	label_total = np.array([0, 0, 0, 0, 0, 0, 0, 0])
	with torch.no_grad():
		for i in range(len(all_inputs)):
			inputs = all_inputs[i]
			labels = all_labels[i]
			label = labels.item()
			outputs = net.forward(inputs)
			_, predicted = torch.max(outputs, 1)
			if (predicted == label):
				correct += 1
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

sequence_length = 10
input_size = 12
hidden_size = num_of_labels
categories = num_of_labels
learning_rate = 0.001
num_of_epochs = 30
current_loss = 0
dataset = "pose-3"

rnn = RNN(input_size, hidden_size, categories)
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

def train(category_tensor, line_tensor):
	hidden = rnn.initHidden()
	optimizer.zero_grad()

	#print(category_tensor)
	#print(line_tensor)
	#sys.exit(-1)

	for i in range(line_tensor.size()[0]):
		output, hidden = rnn(line_tensor[i], hidden)

	loss = criterion(output, category_tensor)
	loss.backward()

	optimizer.step()

	return output, loss.item()

def predict(line_tensor):
	hidden = rnn.initHidden()

	output = 0
	for i in range(line_tensor.size()[0]):
		output, hidden = rnn(line_tensor[i], hidden)

	return output


def get_sequences(labels):

	train_entries = []

	for label in labels:
		entries = get_body_entries(dataset, label)

		for i in range(sequence_length - 1, len(entries)):
			frames = []
			print('---')
			for j in range(i - (sequence_length - 1), i + 1):
				print('J: ' +str(j))
				frames.append([(entries[j].angles * 2) - 1])
			tensor = torch.tensor(frames)
			tensor = tensor.float()
			label = torch.tensor([label_to_number(entries[j].label)])
			train_entries.append(TrainEntry(label, tensor))
	return train_entries



train_entries = get_sequences(body_labels)
random.shuffle(train_entries)

for epoch in range(0, num_of_epochs):
	
	for train_entry in train_entries:
		label = train_entry.label
		tensor = train_entry.tensor
		#print(tensor)
		output, loss = train(label, tensor)
		current_loss += loss

	print(current_loss)
	current_loss = 0




correct = 0
total = len(train_entries)
label_correct = np.zeros(num_of_labels)
label_total = np.zeros(num_of_labels)
with torch.no_grad():
	for train_entry in train_entries:
		label = train_entry.label
		tensor = train_entry.tensor
		outputs = predict(tensor)
		_, predicted = torch.max(outputs, 1)
		if (predicted == label):
			correct += 1
		if (predicted == label):
			label_correct[label] += 1
		label_total[label] += 1

percentage = 100 * correct / total
print('Accuracy of the network on the ' + str(total) + ' test images: %.1f %%' % percentage)
for i in range(num_of_labels):
	print('Accuracy of %s (%d): %d %%' % (body_labels[i], label_total[i], 100 * label_correct[i] / label_total[i]))





#print(loss)

#tensor1 = torch.tensor([[[1,1,1]], [[0.5,0.5,0.5]], [[0,0,0]]])
#tensor1 = tensor1.float()
#label1 = torch.tensor([0])

#tensor2 = torch.tensor([[[-1,-1,-1]], [[-0.5,-0.5,-0.5]], [[0,0,0]]])
#tensor2 = tensor2.float()
#label2 = torch.tensor([1])






