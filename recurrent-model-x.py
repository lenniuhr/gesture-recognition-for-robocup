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

PATH = "./pytorch-model.pth"

os.environ["KMP_DUPLICATE_LIB_OK"]="True"

body_labels = ["spin", "clap", "time-out", "dance", "idle", "fold"]

num_of_labels = len(body_labels)

sequence_length = 10

input_size = 16

hidden_size = 20

recurrent_size = 20

categories = num_of_labels

learning_rate = 0.001

num_of_epochs = 25

dataset = "pose-4"

k = 4

class TrainEntry:
	def __init__(self, label, tensor, frame_nr):
		self.label = label
		self.tensor = tensor
		self.frame_nr = frame_nr

def label_to_number(label):
	return body_labels.index(label)

class RNN(nn.Module):
	def __init__(self, input_size, recurrent_size, hidden_size, output_size):
		super(RNN, self).__init__()

		self.recurrent_size = recurrent_size

		self.r1 = nn.Linear(input_size + recurrent_size, recurrent_size)

		self.fc1 = nn.Linear(input_size + recurrent_size, hidden_size)
		self.fc3 = nn.Linear(hidden_size, num_of_labels)
		self.softmax = nn.LogSoftmax(dim=1)

	def forward(self, x, recurrent):

		combined = torch.cat((x, recurrent), 1)

		recurrent = self.r1(combined)

		x = F.relu(self.fc1(combined))
		x = self.fc3(x)
		x = self.softmax(x)

		return x, recurrent

	def initHidden(self):
		return Variable(torch.zeros(1, self.recurrent_size))

       
print("---------- START ----------")


current_loss = 0

#rnn = RNN(input_size, hidden_size, categories)
#optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)
#criterion = nn.NLLLoss()

def train(model, category_tensor, line_tensor):
	hidden = model.initHidden()
	optimizer.zero_grad()

	#print(category_tensor)
	#print(line_tensor)
	#sys.exit(-1)

	for i in range(line_tensor.size()[0]):
		output, hidden = model(line_tensor[i], hidden)

	loss = criterion(output, category_tensor)
	loss.backward()

	optimizer.step()

	return output, loss.item()

def predict(model, line_tensor):
	hidden = model.initHidden()

	output = 0
	for i in range(line_tensor.size()[0]):
		output, hidden = model(line_tensor[i], hidden)

	return output

def get_sequences(labels):

	train_entries = []

	for label in labels:
		entries = get_body_entries(dataset, label)

		for i in range(0, len(entries), sequence_length):
			frames = []
			for j in range(i, i + sequence_length):
				print(entries[j].label + " - " + str(entries[j].frame_nr))
				frames.append([(entries[j].angles * 2) - 1])
			tensor = torch.tensor(frames)
			tensor = tensor.float()
			label = torch.tensor([label_to_number(entries[j].label)])
			train_entries.append(TrainEntry(label, tensor, entries[j].frame_nr))






	#print(len(train_entries))
	#for entry in train_entries:
	#	print(entry.label)

	#i = 0
	#for entry in train_entries:
	#	i = i + 1
	#	if i % 2 == 0:
	#		entry.label = torch.tensor([label_to_number("spin")])
	#	else:
	#		entry.label = torch.tensor([label_to_number("clap")])

	#for entry in train_entries:
	#	print(entry.label)
	#print(len(train_entries))




	return np.array(train_entries)

def train_model(entries):
	model = RNN(input_size, recurrent_size, hidden_size, categories)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=learning_rate)	

	for epoch in range(num_of_epochs):
		# shuffle entries and create batches
		random.shuffle(entries)
		running_loss = 0
		for i in range (len(entries)):
			tensor = entries[i].tensor
			label = entries[i].label

			hidden = model.initHidden()
			optimizer.zero_grad()

			for i in range(tensor.size()[0]):
				output, hidden = model(tensor[i], hidden)

			loss = criterion(output, label)
			loss.backward()
			optimizer.step()		

			running_loss += loss.item()

		print("Epoch %i: %f loss" % (epoch, running_loss))

	return model

def validate_model():

	entries = get_sequences(body_labels)
	kf = KFold(n_splits = k, shuffle = True)
	percentage_acc = 0
	label_correct_acc = np.zeros(num_of_labels)
	label_total_acc = np.zeros(num_of_labels)
	for train_index, test_index in kf.split(entries):

		train_entries = entries[train_index]
		test_entries = entries[test_index]
		print(len(train_entries))
		print(len(test_entries))


		model = train_model(train_entries)
		percentage, label_correct, label_total = print_results(model, test_entries)
		percentage_acc += percentage
		label_correct_acc += label_correct
		label_total_acc += label_total

	for i in range(num_of_labels):
		print(body_labels[i])
		print(label_total_acc[i])
		print(label_correct_acc[i])
		print("Overall accuracy of %s (%d): %d %%" % (body_labels[i], label_total_acc[i], 100 * label_correct_acc[i] / label_total_acc[i]))
	print("Overall accuracy of the network: %.1f %%" % (percentage_acc / k))

def print_results(model, entries):

	correct = 0
	total = len(entries)
	label_correct = np.zeros(num_of_labels)
	label_total = np.zeros(num_of_labels)
	with torch.no_grad():
		for entry in entries:
			tensor = entry.tensor
			label = entry.label
			label = label.item()

			outputs = predict(model, tensor)
			_, predicted = torch.max(outputs, 1)
			if (predicted == label):
				correct += 1
			else:
				print("Label: " + body_labels[label] + " - " + str(entry.frame_nr) + ", predicted: " + body_labels[predicted])
			if (predicted == label):
				label_correct[label] += 1
			label_total[label] += 1

	percentage = 100 * correct / total
	#print("Accuracy of the network on the " + str(total) + " test images: %.1f %%" % percentage)
	#for i in range(8):
		#print("Accuracy of %s (%d): %d %%" % (body_labels[i], label_total[i], 100 * label_correct[i] / label_total[i]))
	return percentage, label_correct, label_total




#train_entries = get_sequences(body_labels)
#rnn = train_model(train_entries)

#entries = get_sequences(body_labels)
#sys.exit(-1)
validate_model()

#entries = get_sequences(body_labels)




