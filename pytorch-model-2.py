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

from fileutil import *

PATH = './pytorch-model.pth'

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(6, 6)
		self.fc2 = nn.Linear(6, 6)
		self.fc3 = nn.Linear(6, 8)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

# arm up right
# arm up left
# dab
# clap
# idle
# show-left
# show-right

def get_dataset():
	entries = get_all_body_entries()

	random.seed(a=1, version=2)
	random.shuffle(entries)

	angles = np.array(list(map(lambda e: e.angles, entries)))
	angles = (angles * 2) - 1
	labels = np.array(list(map(lambda e: label_to_number(e.label), entries)))

	tensor_x = torch.from_numpy(angles) # transform to torch tensor
	tensor_y = torch.from_numpy(labels)

	return torch.utils.data.TensorDataset(tensor_x,tensor_y) # create your datset

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

classes = ['idle', 'tpose', 'dab', 'clap', 'show-left', 'show-right', 'arm-up-left', 'arm-up-right']

#-------- IMPORT IMAGES --------#

print("---------- START ----------")

net = Net()
net = net.double()

dataset = get_dataset()
print("Dataset size: " + str(len(dataset)))
trainloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

print('Start Training')
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
for epoch in range(15):
	running_loss = 0.0
	for i, data in enumerate(trainloader, 0):

		inputs, labels = data
		optimizer.zero_grad()
		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		# print statistics
		running_loss += loss.item()
		if i % 70 == 69:    # print every 70 mini-batches
			print('[%d, %5d] loss: %.3f' %
				  (epoch + 1, i + 1, running_loss / 2000))
			running_loss = 0.0
print('Finished Training')

#---------- VALIDATION --------#

entries = get_all_body_entries()

correct = 0
total = len(entries)
class_correct = [0, 0, 0, 0, 0, 0, 0, 0]
class_total = [0, 0, 0, 0, 0, 0, 0, 0]
with torch.no_grad():
	for entry in entries:
		inputs = torch.from_numpy(np.array([entry.angles]))
		inputs = (inputs * 2) - 1
		label = torch.from_numpy(np.array([label_to_number(entry.label)]))

		outputs = net(inputs)
		_, predicted = torch.max(outputs.data, 1)
		#print('Predicted: ' + str(predicted))
		correct += (predicted == label)
		class_correct[label] += (predicted == label)
		class_total[label] += 1

print('Accuracy of the network on the ' + str(total) + ' test images: %d %%' % (100 * correct / total))


for i in range(8):
	print('Accuracy of %s (%d): %d %%' % (classes[i], class_total[i], 100 * class_correct[i] / class_total[i]))






