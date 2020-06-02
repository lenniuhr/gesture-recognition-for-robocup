

def hyperopt():

	learning_rates = [0.001] #, 0.005, 0.01]
	batch_sizes = [1] #, 25, 210]
	learning_rate = 0.001
	batch_size = 
	hidden_sizes = [5, 10, 20]
	nums_layers = [1, 2, 3]

	results = []

	for hidden_size in hidden_sizes:
		for num_layers in nums_layers:
			iteration = "num_layers: %d, hidden_size: %d"  % (num_layers, hidden_size)
			print(iteration)

			accuracy_acc = []
			for i in range (0, 1):
				accuracys = validate_model(hidden_size, num_layer, learning_rate, batch_size)
				accuracy_acc.extend(accuracys)

			mean = np.mean(np.array(accuracy_acc))
			std = np.std(np.array(accuracy_acc))

			results.append(iteration)
			results.append("Mean: %.2f, std: %.2f" % (mean, std))

	print("---------- RESULTS ----------")
	for result in results:
		print(result)