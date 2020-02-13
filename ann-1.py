# Jacob Kroner
# Neural Networks

import sys
import random
import math

# Neuron class
class Neuron:

	# Initializes all weights, including bias weight
	def __init__(self, num_inputs):
		weights = []
		for i in range(num_inputs + 1):
			weight = random.uniform(0.001, 0.5)
			weights.append(weight)
		self.weights = weights

	# Calculates the sum of weights * inputs
	def calculate_weight(self, data):
		bias = 1
		bias_weight = self.weights[0]
		sum = 0
		sum += bias * bias_weight
		for i in range(len (data)):
			sum += float(data[i]) * self.weights[i+1]
		return sum

	# Activation function
	def activation(self, sum):
		return float(1) / float(1 + math.exp(-1 * sum))

	# Processes the neuron
	def process(self, data):
		sum = self.calculate_weight(data)

		activation_num = self.activation(sum)

		return activation_num

	def get_weights(self):
		return self.weights

	def update_weights(self, weights):
		self.weights = weights

# Neural Network class
class Neural_Network:

	input_layer = []
	hidden_layers = []
	output_layer = []

	# Since our problem has three flower types, 3 outputs needed
	output_layer_size = 3

	# Initializes input layer
	def init_input_layer(self, layer_size, num_inputs):
		for i in range(layer_size):
			neuron = Neuron(num_inputs)
			self.input_layer.append(neuron)

	# Initializes the hidden layers
	def init_hidden_layers(self, num_hidden):
		for i in range (num_hidden):
			hidden_layer = []
			for j in range(len (self.input_layer)):
				if i == 0:
					num_inputs = len (self.input_layer)
				else:
					num_inputs = len (self.hidden_layers[-1])

				neuron = Neuron(num_inputs)
				hidden_layer.append(neuron)
			self.hidden_layers.append(hidden_layer)


	# Initializes the output layer
	def init_output_layer(self):
		for i in range(self.output_layer_size):
			num_inputs = len (self.hidden_layers[-1])
			neuron = Neuron(num_inputs)
			self.output_layer.append(neuron)

	# Initializes all
	def __init__(self, layer_size, num_inputs, num_hidden):
		self.init_input_layer(layer_size, num_inputs)
		self.init_hidden_layers(num_hidden)
		self.init_output_layer()

	# Calculate and return the outputs for each neuron in the input layer
	def get_input_layer_outputs(self, input_layer, data):
		input_layer_outputs = []
		for neuron in input_layer:
			input_layer_outputs.append(neuron.process(data))
		return input_layer_outputs

	# Calculate and return the outputs for each neuron in each of the hidden layers
	def get_hidden_layers_outputs(self, hidden_layers, input_layer_outputs):
		hidden_layers_outputs = []
		i = 0
		for hidden_layer in self.hidden_layers:
			output = []
			for neuron in hidden_layer:
				if i == 0:
					output.append(neuron.process(input_layer_outputs))
				else:
					output.append(neuron.process(hidden_layers_outputs[-1]))

			hidden_layers_outputs.append(output)
			i += 1
		return hidden_layers_outputs

	# Calculate and return the outputs for each neuron in the output layer
	def get_output_layer_outputs(self, output_layer, hidden_layers_outputs):
		output_layer_outputs = []
		for neuron in self.output_layer:
			output_layer_outputs.append(neuron.process(hidden_layers_outputs[-1]))
		return output_layer_outputs

	# Gets the error on an individual example
	def get_errors(self, output_layer_outputs, answer):
		error = [0, 0, 0]
		if answer == "Iris-setosa\n":
			for i in range(len (output_layer_outputs)):
				if i == 0:
					error[i] = 1 - output_layer_outputs[i]
				else:
					error[i] = 0 - output_layer_outputs[i]

		elif answer == "Iris-versicolor\n":
			for i in range(len (output_layer_outputs)):
				if i == 1:
					error[i] = 1 - output_layer_outputs[i]
				else:
					error[i] = 0 - output_layer_outputs[i]

		elif answer == "Iris-virginica\n":
			for i in range(len (output_layer_outputs)):
				if i == 2:
					error[i] = 1 - output_layer_outputs[i]
				else:
					error[i] = 0 - output_layer_outputs[i]
		else:
			raise Exception("Flower type not recognized")

		return error

	# Calculates and returns the deltaJ values for the output layer
	def get_deltaJs(self, output_layer_outputs, errors):
		deltaJs = []
		for i in range(len (output_layer_outputs)):
			deltaJs.append(output_layer_outputs[i] * (1 - output_layer_outputs[i]) * errors[i])
		return deltaJs

	# Calculates and reutrns the deltaI values for the hidden layers
	def get_hidden_deltaIs(self, hidden_layers, hidden_layers_outputs, output_layer, deltaJs):
		deltaIs = []
		for i in range(len (hidden_layers) - 1, -1, -1):
			hidden_layer = hidden_layers[i]
			deltaIs_in_layer = []
			j = 0
			for neuron in hidden_layer:
				if i == len (hidden_layers) - 1:
					weight_sum = 0
					k = 0
					for output_neuron in output_layer:
						weight_sum += output_neuron.get_weights()[j+1] * deltaJs[k]
						k += 1
					deltaI = hidden_layers_outputs[i][j] * (1 - hidden_layers_outputs[i][j]) * weight_sum
				else:
					weight_sum = 0
					k = 0
					for hidden_neuron in hidden_layers[i-1]:
						weight_sum += hidden_neuron.get_weights()[j+1] * deltaIs[-1][k]
						k += 1
					deltaI = hidden_layers_outputs[i][j] * (1 - hidden_layers_outputs[i][j]) * weight_sum
				deltaIs_in_layer.append(deltaI)
				j += 1
			deltaIs.append(deltaIs_in_layer)
		return deltaIs

	# Calculates and returns the deltaI values for the input layer
	def get_input_deltaIs(self, input_layer, input_layer_outputs, hidden_layers, hidden_deltaIs):
		input_layer_deltaIs = []
		j = 0
		for neuron in input_layer:
			weight_sum = 0
			k = 0
			for hidden_neuron in hidden_layers[0]:
				weight_sum += hidden_neuron.get_weights()[j+1] * hidden_deltaIs[-1][k]
				k += 1
			input_deltaI = input_layer_outputs[j] * (1 - input_layer_outputs[j]) * weight_sum
			input_layer_deltaIs.append(input_deltaI)
			j += 1

		return input_layer_deltaIs

	# Updates the weights of output layer neurons using the deltaJ values
	def update_output_weights(self, output_layer, hidden_layers_outputs, deltaJs, alpha):
		index = 0
		for neuron in output_layer:
			old_weights = neuron.get_weights()
			new_weights = []

			deltaJ = deltaJs[index]

			j = 0
			output_num = 1
			for weight in old_weights:
				# Bias term
				if j == 0:
					ouput_num = 1
				else:
					output_num = hidden_layers_outputs[-1][j - 1]

				new_weights.append(weight + alpha * output_num * deltaJ)
				j += 1

			neuron.update_weights(new_weights)
			index += 1

	# Updates the weights of hidden layer neurons using the hidden layer deltaI values
	def update_hidden_weights(self, hidden_layers, hidden_layers_outputs, input_layer_outputs, hidden_deltaIs, alpha):
		curr_layer = 0
		for layer in hidden_layers:
			index = 0
			deltaIs_curr_layer = hidden_deltaIs[curr_layer]

			for neuron in layer:
				old_weights = neuron.get_weights()
				new_weights = []
				
				deltaI = deltaIs_curr_layer[index]
				j = 0
				output_num = 1
				for weight in old_weights:
					# Bias term
					if j == 0:
						output_num = 1
					else:
						if curr_layer == 0:
							output_num = input_layer_outputs[j - 1]
						else:
							output_num = hidden_layers_outputs[curr_layer - 1][j - 1]
					new_weights.append(weight + alpha * output_num * deltaI)
					j += 1
				neuron.update_weights(new_weights)
				index += 1

			curr_layer += 1

	# Updates the weights of input layer neurons using the input layer deltaI values
	def update_input_weights(self, input_layer, input_deltaIs, data, alpha):
		index = 0
		for neuron in input_layer:
			old_weights = neuron.get_weights()
			new_weights = []

			deltaI = input_deltaIs[index]
			j = 0
			output_num = 1
			for weight in old_weights:
				# Bias term
				if j == 0:
					output_num = 1
				else:
					output_num = data[j - 1]
				new_weights.append(weight + alpha * output_num * deltaI)
				j +=1
			neuron.update_weights(new_weights)
			index += 1

	# Trains the network until stopping criterion is met
	def back_propagation(self, training_data, validation_data):
		train = True
		attempts = 0
		random.shuffle(training_data)
		while train:
			for example in training_data:

				data_strings = example[0:(len (example) - 1)]

				data = [float(i) for i in data_strings]

				answer = example[len (example) - 1]

				input_layer_outputs = self.get_input_layer_outputs(self.input_layer, data)
				hidden_layers_outputs = self.get_hidden_layers_outputs(self.hidden_layers, input_layer_outputs)
				output_layer_outputs = self.get_output_layer_outputs(self.output_layer, hidden_layers_outputs)

				errors = self.get_errors(output_layer_outputs, answer)

				deltaJs = self.get_deltaJs(output_layer_outputs, errors)

				hidden_deltaIs = self.get_hidden_deltaIs(self.hidden_layers, hidden_layers_outputs, self.output_layer, deltaJs)

				input_deltaIs = self.get_input_deltaIs(self.input_layer, input_layer_outputs, self.hidden_layers, hidden_deltaIs)
				
				alpha = 0.1

				self.update_output_weights(self.output_layer, hidden_layers_outputs, deltaJs, alpha)

				self.update_hidden_weights(self.hidden_layers, hidden_layers_outputs, input_layer_outputs, hidden_deltaIs, alpha)

				self.update_input_weights(self.input_layer, input_deltaIs, data, alpha)
			print("Statistics for validation data on attempt " + str(attempts))
			per_corr = self.data_statistics(validation_data)
			attempts += 1
			if per_corr > 0.97 or attempts > 10000:
				train = False 

	# Runs an individual example through the network
	def run_example(self, example):
		data_strings = example[0:(len (example) - 1)]

		data = [float(i) for i in data_strings]

		answer = example[len (example) - 1]

		input_layer_outputs = self.get_input_layer_outputs(self.input_layer, data)
		hidden_layers_outputs = self.get_hidden_layers_outputs(self.hidden_layers, input_layer_outputs)
		output_layer_outputs = self.get_output_layer_outputs(self.output_layer, hidden_layers_outputs)

		if answer == "Iris-setosa\n":
			return output_layer_outputs[0] > output_layer_outputs[1] and output_layer_outputs[0] > output_layer_outputs[2]
		elif answer == "Iris-versicolor\n":
			return output_layer_outputs[1] > output_layer_outputs[0] and output_layer_outputs[1] > output_layer_outputs[2]
		elif answer == "Iris-virginica\n":
			return output_layer_outputs[2] > output_layer_outputs[0] and output_layer_outputs[2] > output_layer_outputs[1]					
		else:
			raise Exception("Flower type not recognized")

	# Calculates accuracy on the testing data
	def data_statistics(self, testing):
		correct_count = 0
		wrong_count = 0
		for example in testing:
			if self.run_example(example):
				correct_count += 1
			else:
				wrong_count += 1

		per_corr = float(correct_count) / float(correct_count + wrong_count)
		print("Correct count: " + str(correct_count))
		print("Wrong count: " + str(wrong_count))
		print("Percent correct: " + str(per_corr) + "\n")
		return per_corr


def main():

	file = sys.argv[1]
	f = open(file, "r")
	lines = f.readlines()

	training = []
	validation = []
	testing = []

	examples_data = []

	# Normalizes the data in the range 0-1
	for i in range(len(lines)):
		if lines[i] != "\n":
			lines[i] = lines[i].split(',')
			data_strings = lines[i][0:(len (lines[i]) - 1)]
			data = [float(i) for i in data_strings]

			examples_data.append(data)

	# Variance method, values gotten from Iris Description

	mean_sepal_length = 0.83
	mean_sepal_width = 0.43
	mean_petal_length = 1.76
	mean_petal_width = 0.76

	std_dev_sepal_length = 0.7826
	std_dev_sepal_width = -0.4194
	std_dev_petal_length = 0.9490
	std_dev_petal_width = 0.9565
	for i in range(len(examples_data)):
		examples_data[i][0] = (examples_data[i][0] - mean_sepal_length) / (std_dev_sepal_length)
		examples_data[i][1] = (examples_data[i][1] - mean_sepal_width) / (std_dev_sepal_width)
		examples_data[i][2] = (examples_data[i][2] - mean_petal_length) / (std_dev_petal_length)
		examples_data[i][3] = (examples_data[i][3] - mean_petal_width) / (std_dev_petal_width)

	num = random.randint(0, 1)

	# Training - 50%, Validation - 25%, Testing - 25%
	for i in range(len (lines)):
		if lines[i] != "\n":
			if i % 4 == num or i % 4 == num + 1:
				training.append(examples_data[i] + [lines[i][-1]])
			elif i % 4 == num + 2:
				validation.append(examples_data[i] + [lines[i][-1]])
			else:
				testing.append(examples_data[i] + [lines[i][-1]])
	neural_network = Neural_Network(4, 4, 1)
	neural_network.back_propagation(training, validation)

	print("Statistics for testing data: ")
	neural_network.data_statistics(testing)


main()