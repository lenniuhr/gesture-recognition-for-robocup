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

body_labels = ['idle', 'tpose', 'dab', 'clap', 'show-left', 'show-right', 'arm-up-left', 'arm-up-right']

def label_to_number(label):
	if(label == 'idle'):
		return 0
	if(label == 'tpose'):
		return 1
	if(label == 'dab'):
		return 2
	if(label == 'clap'):
		return 3
	if(label == 'show-left'):
		return 4
	if(label == 'show-right'):
		return 5
	if(label == 'arm-up-left'):
		return 6
	if(label == 'arm-up-right'):
		return 7

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
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
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

input_size = 3
hidden_size = 10
categories = 2
learning_rate = 0.005
current_loss = 0

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

tensor1 = torch.tensor([[[1,1,1]], [[0.5,0.5,0.5]], [[0,0,0]]])
tensor1 = tensor1.float()
label1 = torch.tensor([0])

tensor2 = torch.tensor([[[-1,-1,-1]], [[-0.5,-0.5,-0.5]], [[0,0,0]]])
tensor2 = tensor2.float()
label2 = torch.tensor([1])

for epoch in range(0,10):
	for i in range(0,100):

		output, loss = train(label1, tensor1)
		output, loss = train(label2, tensor2)
		current_loss += loss
	print(current_loss)
	current_loss = 0

#print(loss)






