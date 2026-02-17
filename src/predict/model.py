import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class WeatherPredictModel(nn.Module):
	def __init__(self):
		super(WeatherPredictModel, self).__init__()

		self.lin = nn.Linear(3, 2)


	def forward(self, x):
		return self.lin(x)