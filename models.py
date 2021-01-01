#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class MLP(nn.Module):
	def __init__(self, input_dim, output_dim, hidden_dims=[], activation='relu', p_drop=0, batchnorm='False'):
		super(MLP, self).__init__()

		# layer-dimensions
		dlayers = [input_dim] + hidden_dims + [output_dim]

		# network definintion
		self.network = nn.Sequential()
		l=0
		for i in range(len(dlayers) - 2):
			self.network.add_module(
					name=str(l),
					module=nn.Linear(dlayers[i], dlayers[i + 1]))
			self.network.add_module(
					name=str(l + 1),
					module=nn.ReLU(inplace=True) if activation=='relu' else 
					(nn.Tanh() if activation=='tanh' else nn.Sigmoid()))
			self.network.add_module(
					name=str(l + 2),
					module=nn.Dropout(p_drop))
			if batchnorm:
				self.network.add_module(
					name=str(l + 3),
					module=nn.BatchNorm1d(dlayers[i + 1]))
				l += 1
			l += 3

		# last layer
		self.network.add_module(
				name=str(l),
				module=nn.Linear(dlayers[-2], dlayers[-1]))

	def forward(self, x):
		return self.network(x)

class CNN(nn.Module):
	def __init__(self, arch, channels, kernels, out, input_shape, # cnn parameters
				 hidden_dims=[], activation='relu', p_drop=0, batchnorm='False' # mlp parameters
				 ):
		super(CNN, self).__init__()
		self.features = nn.Sequential()

		i, j , l = 0, 0, 0
		for layer in arch:
			if layer == 'conv':
				self.features.add_module(
				name=str(l),
				module=nn.Conv2d(channels[i], channels[i + 1], kernel_size=kernels[j], stride=1, padding=1))
				i += 1
				j += 1
			elif layer == 'maxpool':
				self.features.add_module(
				name=str(l),
				module=nn.MaxPool2d(kernel_size=kernels[j], stride=2, padding=0))
				j +=1
			else:
				self.features.add_module(
				name=str(l),
				module=nn.ReLU(inplace=True) if layer=='relu' else 
					  (nn.Tanh() if layer=='tanh' else nn.Sigmoid()))    
			l += 1

		flt_dim = self.flattendim(input_shape)
		self.linear = MLP(flt_dim, out, hidden_dims, activation, p_drop, batchnorm) # number 128 needs to be found
    
	def forward(self, x):
		x = self.features(x)
		#         return x
		x = x.view((len(x), -1))
		return self.linear(x)

	def flattendim(self, input_shape):
		for module in self.features:
			name = module.__class__.__name__
			if name == 'Conv2d':
				(cin, hin, win) = input_shape
				cout = module.out_channels
				hout = int(np.floor((hin + 2 * module.padding[0] - module.dilation[0] * (module.kernel_size[0] - 1) - 1) / module.stride[0] + 1))
				wout = int(np.floor((win + 2 * module.padding[1] - module.dilation[1] * (module.kernel_size[1] - 1) - 1) / module.stride[1] + 1))
				input_shape = (cout, hout, wout)
			elif name == 'MaxPool2d':
				(cin, hin, win) = input_shape
				cout = cin
				hout = int(np.floor((hin + 2 * module.padding - module.dilation * (module.kernel_size - 1) - 1) / module.stride + 1))
				wout = int(np.floor((win + 2 * module.padding - module.dilation * (module.kernel_size - 1) - 1) / module.stride + 1))
				input_shape = (cout, hout, wout)

		return int(np.prod(np.array(input_shape)))


class Data(Dataset):
	def __init__(self, dataset, features):
		self.len = len(dataset)
		self.x = torch.from_numpy(dataset.iloc[:, :features].values).float()
		self.y = torch.from_numpy(dataset.iloc[:, features].values).long()

	def __getitem__(self, index):
		return self.x[index], self.y[index]

	def __len__(self):
		return self.len
