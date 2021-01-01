#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from torch import optim
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from helper import *

import torch.nn as nn
from models import CNN, Data

'''dataset related parameters'''
file_name = "iris_dataset.txt"
features = 4
num_class = 10

'''CNN related parameters'''
conv, maxpool, relu, tanh, sigmoid, = 'conv', 'maxpool', 'relu', 'tanh', 'sigmoid'  # possible layers
arch = [conv, relu, conv, relu, maxpool, conv, relu, maxpool]	# layerwise architecture
channels = [1, 32, 64, 64]		# equal to number of conv+1 layers
kernels = [3, 3, 2, 5, 2]		# square kernels equal to number of conv+maxpool layers
input_shape = (1, 28, 28)

'''MLP related parameters'''
hidden_dims = [6, 5]	# hidden nodes in hidden layers
p_drop = 0				# dropout probability
activation = 'relu' 	# activation function
epochs = 15				# number of epochs
batchnorm = True

'''load and split the dataset'''
trainer = datasets.MNIST(root = "data", 
					   train=True,
					   download=True,
					   transform=transforms.ToTensor())
tester = datasets.MNIST(root = "data", 
					   train=False,
					   download=False,
					   transform=transforms.ToTensor())

'''model setup'''
model = CNN(arch, channels, kernels, num_class, input_shape, #cnn parameters
			hidden_dims, activation, p_drop, batchnorm # mlp parameters
			)
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

print(f'The Model:\n{model}')

'''train the model'''
(train_loss, train_acc, test_loss, test_acc) =\
				 model_train(model,
							  trainer,
							  optimizer,
							  criterion,
							  tester=tester,
							  batch_size=500,
							  epochs=epochs)

'''plot the performance'''
performance_plot(train_loss, test_loss, 0.7, "loss", "Loss.jpeg")
performance_plot(train_acc, test_acc, 0.7, "accuracy", "Accuracy.jpeg")

'''test output'''
confusion_matrix_plot(model, tester, "Conf_Matrix.jpeg")