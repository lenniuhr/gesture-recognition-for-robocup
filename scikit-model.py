import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split, KFold
import sys

from fileutil import *

classes = ['idle', 'tpose', 'dab', 'clap', 'show-left', 'show-right', 'arm-up-left', 'arm-up-right']

def score_body_model():
	dataset = get_all_body_entries()
	print("Dataset size: " + str(len(dataset)))

	labels = []
	angles = []

	for i in range(0, len(dataset)):
	    labels.append(dataset[i].label)
	    angles.append(dataset[i].angles)

	y = np.array(labels)
	X = np.array(angles)
	print(angles[0])
	k = 5
	score = 0
	kf = KFold(n_splits = k, shuffle = True)
	print("------ Scores: ------")
	for train_index, test_index in kf.split(X):
		X_test = X[test_index]
		X_train = X[train_index]

		y_test = y[test_index]
		y_train = y[train_index]

		model = MLPClassifier(solver = 'adam', max_iter = 2000, alpha = 1e-4, hidden_layer_sizes = (12, 12), random_state = 0)
		print(model)
		#sys.exit(-1)
		model.fit(X_train, y_train)

		for c in classes:
			class_entries = get_body_entries(c)
			class_labels = []
			class_angles = []
			for i in range(0, len(class_entries)):
			    class_labels.append(class_entries[i].label)
			    class_angles.append(class_entries[i].angles)
			y_class = np.array(class_labels)
			X_class = np.array(class_angles)
			print("Class %s: %f" % (c, model.score(X_class, y_class)))

		print("KFold: %f" % (model.score(X_test, y_test)))

		score += model.score(X_test, y_test)
	print("Overall score: " + str(score / k))

def predict_file(category, file, model):
	entry = get_entry(file)
	print("File: " + file)
	print("Label: " + entry['label'])
	print("Predicted: " + str(model.predict([entry['angles']])))

#------------- START -----------------

model = score_body_model()

#files = ["images/right-hand-one-5.jpeg", "images/left-hand-four-5.jpeg"]
#predict_file(files, model)


