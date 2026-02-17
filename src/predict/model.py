import torch
import torch.nn as nn


class WeatherPredictModel(nn.Module):
    def __init__(self):
        super(WeatherPredictModel, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        
        self.lstm = nn.LSTM(input_size=256, hidden_size=128, num_layers=2, batch_first=True, bidirectional=False)
        
        self.temp_fc1 = nn.Linear(128, 64)
        self.temp_fc2 = nn.Linear(64, 24)
        
        self.precip_fc1 = nn.Linear(128, 64)
        self.precip_fc2 = nn.Linear(64, 24)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.relu(self.conv3(x))
        
        x = x.permute(0, 2, 1)
        
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        
        temp_out = self.relu(self.temp_fc1(x))
        temp_out = self.temp_fc2(temp_out)
        
        precip_out = self.relu(self.precip_fc1(x))
        precip_out = self.sigmoid(self.precip_fc2(precip_out))
        
        return temp_out, precip_out