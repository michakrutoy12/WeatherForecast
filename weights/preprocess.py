import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pickle
import glob
import os
import re

train_columns = ['T', 'P0', 'Dp']
target_columns = ['T', 'Pr']
HISTORY_LEN = 24
PREDICTION_LEN = 24
TRAIN_SPLIT = 0.9

precipitation_map = {
    'ливень': 8, 'сильный ливень': 9, 'вблизи гроза': 5,
    'небольшой град и/или снежная крупа': 8, 'сильный снег': 8, 
    'буря снег': 10, 'слабый ливень': 4, 'град': 9,
    'слабый снег': 3, 'дождь': 7, 'слабый гроза': 6,
    'вблизи ливень': 3, 'морось': 4, 'гроза': 8,
    'слабый морось': 3, 'слабый дождь': 5, 'сильный гроза': 10,
    'слабый замерзающий дождь': 8, 'снег': 7
}

def score_weather(weather_string):
    if pd.isna(weather_string) or weather_string == '':
        return 0.0
    parts = str(weather_string).lower().split(',')
    scores = []
    for p in parts:
        clean_p = re.sub(r'\s*\([^)]*\)', '', p).strip().replace(')', '').replace('(', '')
        scores.append(precipitation_map.get(clean_p, 0))
    output = (sum(scores) / len(scores)) / 10 if scores else 0.0
    return 1 if output > 0 else 0

all_files = glob.glob(os.path.join('dataset', "*.csv"))
dataframes = []

for filename in all_files:
    try:
        df = pd.read_csv(filename, sep=';', encoding='utf-8', quotechar='"', skiprows=1, index_col=False)
        dataframes.append(df)
    except Exception as e:
        print(f"Ошибка чтения файла {filename}: {e}")

full_df = pd.concat(dataframes, ignore_index=True)

full_df['D'] = pd.to_datetime(full_df['D'], dayfirst=True)
full_df = full_df.sort_values(by='D').set_index('D')

full_df['T'] = full_df['T'].rolling(window=6, min_periods=1).mean()
full_df['Pr'] = full_df['Pr'].apply(score_weather)

a = 17.27
b = 237.7
# full_df['H'] = 100 * np.exp((a * full_df['Dp']) / (b + full_df['Dp'])) / np.exp((a * full_df['T']) / (b + full_df['T']))
# full_df['Dp'] = full_df['H'].round(2)

columns_to_keep = list(set(train_columns + target_columns))
full_df = full_df[columns_to_keep].resample('1h').mean()
dataset = full_df.interpolate(method='linear')

train_size = int(len(dataset) * TRAIN_SPLIT)
train_data = dataset.iloc[:train_size]

stats = {}
for col in train_columns:
    if col == 'Pr': continue
    
    mean = train_data[col].mean()
    std = train_data[col].std()
    
    if std == 0: std = 1e-6 
    
    dataset[col] = (dataset[col] - mean) / std
    stats[col] = {'mean': mean, 'std': std}

inputs = dataset[train_columns].values
targets = dataset[target_columns].values

x_windows = sliding_window_view(inputs, window_shape=(HISTORY_LEN, inputs.shape[1]))
x_windows = x_windows[:-(PREDICTION_LEN), 0] 

y_windows = sliding_window_view(targets, window_shape=(PREDICTION_LEN, targets.shape[1]))
y_windows = y_windows[HISTORY_LEN:, 0]

X = x_windows.transpose(0, 2, 1)
Y = y_windows.transpose(0, 2, 1)


# output = {
#     'train_columns': train_columns,
#     'target_columns': target_columns,
#     'train': {
#         'X': X[:train_size],
#         'y_temp': Y[:train_size, 0, :],
#         'y_precip': Y[:train_size, 1, :]
#     },
#     'test': {
#         'X': X[train_size:],
#         'y_temp': Y[train_size:, 0, :],
#         'y_precip': Y[train_size:, 1, :]
#     },
#     'stats': stats,
#     'history_length': HISTORY_LEN,
#     'prediction_length': PREDICTION_LEN,
#     'train_split': TRAIN_SPLIT
# }

output = {
    'X': X,
    'y': Y[:, 0, :],
    'y_precip': Y[:, 1, :],
    'T_mean': stats['T']['mean'],
    'T_std': stats['T']['std'],
    'U_mean': stats['Dp']['mean'],
    'U_std': stats['Dp']['std'],
    'P_mean': stats['P0']['mean'],
    'P_std': stats['P0']['std']
}

with open('datasetGRU.pkl', 'wb') as file:
    pickle.dump(output, file)

print(f"Dataset prepared. Splitted at index {train_size}/{len(dataset)}")