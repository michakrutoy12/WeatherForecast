import numpy as np
import torch
from src.predict.model import WeatherPredictModel
import pickle

with open('weights/dataset/prepared.pkl', 'rb') as file:
	params = pickle.load(file)

model = WeatherPredictModel()