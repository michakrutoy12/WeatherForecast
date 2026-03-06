from src.predict import LSTMInformer

import torch
import pickle

import numpy as np
from datetime import datetime, timedelta


with open('weights/dataset/prepared.pkl', 'rb') as file:
    dataset = pickle.load(file)

del dataset['columns']
del dataset['train']
del dataset['targets']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTMInformer(
    enc_in=8,
    c_out=5,
    out_len=24*5,
    d_model=256,
    n_heads=16,
    n_layers=4,
    factor=5
).to(device)

model.load_state_dict(torch.load('./weights/weights.pth', map_location=device))

def predict(X, device=device):
    model.eval()
    with torch.no_grad():
        tensor_x = torch.from_numpy(X).unsqueeze(0).to(device)
        
        output = model(tensor_x)
        pred_norm = output.squeeze().cpu().numpy()
        
        data = {
            'precipitation': pred_norm[:, 0] * 100,
            'humidity': pred_norm[:, 3] * 100 + 50,
            'pressure': pred_norm[:, 1] * dataset['P_std'] + dataset['P_mean'],
            'temperature': pred_norm[:, 2] * dataset['T_std'] + dataset['T_mean'],
            'wind_speed': np.exp(pred_norm[:, 4] * dataset['W_std'] + dataset['W_mean']) - dataset['W_epsilon'],
        }
        
    return data

def preprocess(start_timestamp, pressure, temperature, humidity):
    now = datetime.fromtimestamp(start_timestamp)
    day = np.array([(now + timedelta(hours=dataset['sampling_interval'] * i)).timetuple().tm_yday for i in range(dataset['prediction_lenght'] // dataset['sampling_interval'])])
    time = np.array([(now + timedelta(hours=dataset['sampling_interval'] * i)).hour for i in range(dataset['prediction_lenght'] // dataset['sampling_interval'])])


    hour_sin = np.sin(2 * np.pi * time / 24)
    hour_cos = np.cos(2 * np.pi * time / 24)
    day_sin = np.sin(2 * np.pi * day / 365.25)
    day_cos = np.cos(2 * np.pi * day / 365.25)

    norm_pressure = (pressure[1:] - dataset['P_mean']) / dataset['P_std']
    norm_temp = (temperature - dataset['T_mean']) / dataset['T_std']

    norm_hum = humidity / 100.0 - 0.5

    pressure_trend = []
    for i in range(1, len(pressure)):
        pressure_trend.append(pressure[i] - pressure[i-1])

    pressure_trend = (np.array(pressure_trend) - dataset['Pt_mean']) / dataset['Pt_std']

    features = np.stack([hour_sin, hour_cos, day_sin, day_cos, norm_pressure, pressure_trend, norm_temp, norm_hum])
    return features.T.astype(np.float32)