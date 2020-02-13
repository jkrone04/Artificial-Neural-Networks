Instructions for use: To run, use the command
	python ann.py "ANN - Iris data.txt"

The neural network uses a sigmoid activation function throughout and has one hidden layer (but can be modified to have more).
It trains on the training set, and ensures it's not overfitting by checking accuracy on the validation set. Finally, once the network is trained you can run the test data through it using the data_statistics function.
