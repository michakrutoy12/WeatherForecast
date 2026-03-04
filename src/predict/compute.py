# from src.predict import LSTMInformer
# import torch
# import pickle

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = LSTMInformer()
# model.load_state_dict(torch.load('./weights/weights.pth', map_location=device))

# with open('weights/dataset/prepared.pkl', 'rb') as file:
# 	dataset = pickle.load(file)

# del dataset['columns']
# del dataset['train']
# del dataset['targets']

# def predict(model, X, device=device):
#     model.eval()
#     with torch.no_grad():
#         tensor_x = torch.from_numpy(X).unsqueeze(0).to(device)
        
#         output = model(tensor_x)
#         pred_norm = output.squeeze().cpu().numpy()
        
#         prediction_celsius = (pred_norm * dataset['T_std']) + dataset['T_mean']
        
#         return prediction_celsius

# def preprocess(day_cos, day_sin, hour_cos, hour_sin, pressure, pressure_trend, temperature, humidity, wind_speed):
#     norm_pressure = (pressure - dataset['P_mean']) / dataset['P_std']
#     norm_trend = (pressure_trend - dataset['Pt_mean']) / dataset['Pt_std']
#     norm_temp = (temperature - dataset['T_mean']) / dataset['T_std']
    
#     norm_wind = (np.log(wind_speed + dataset['W_epsilon']) - dataset['W_mean']) / dataset['W_std']
    
#     norm_hum = humidity / 100.0

#     return np.array([
#         hour_sin, hour_cos, day_sin, day_cos, 
#         precipation, norm_pressure, norm_trend, 
#         norm_temp, norm_hum, norm_wind
#     ], dtype=np.float32)