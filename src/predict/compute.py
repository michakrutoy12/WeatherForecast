from src.predict import WeatherPredictModel
import torch
import pickle

device = torch.device('cpu')
model = WeatherPredictModel()
model.load_state_dict(torch.load('weights/weights.pth', map_location=device))

with open('weights/dataset/prepared.pkl', 'rb') as file:
	dataset = pickle.load(file)

del dataset['columns']
del dataset['train']
del dataset['targets']

def predict(model, X, device=device):
    X_test_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    model.eval()
    
    with torch.no_grad():
        temp_pred, precip_pred = model(X_test_tensor)
    
    return temp_pred.cpu().numpy(), precip_pred.cpu().numpy()

def compute(day_cos, day_sin, hour_cos, hour_sin, pressure, pressure_trend, temperature, humidity, wind_speed):
    T_norm = (np.array(T_raw) - dataset['T_mean']) / dataset['T_std']
    P_norm = (np.array(P_raw) - dataset['P_mean']) / dataset['P_std']
    U_norm = (np.array(U_raw) - dataset['U_mean']) / dataset['U_std']

    output = predict(model, np.array([[T_norm, P_norm, U_norm]]))

    return (output[0][0] * dataset['T_std'] + dataset['T_mean']).tolist(), (output[1][0] * 100).tolist()