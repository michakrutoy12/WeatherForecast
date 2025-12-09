import os
import pickle
import numpy as np
import re
from datetime import datetime

targets = [x for x in os.listdir('dataset') if x.split('.')[-1] == 'csv']

total_read = []

for target in targets:
	with open("dataset/" + target, 'r', encoding='UTF-8') as file:
		for line in file.readlines()[::-1]:
			line = line.replace('"', '').split(';')
			if line[0].split(' ')[1].split(':')[1] not in ('00', '30'):
				continue
			if line[1] == '' or line[2] == '':
				continue
			total_read.append({
				'date': line[0],
				'temperature': line[1],
				'sea_level_pressure': line[2],
				'pressure': line[3],
				'humidity': line[4] or 0,
				'wind': line[5],
				'wind_speed': line[6],
				'max_wind_speed': line[7] or 0,
				'weather': line[8],
				'cloud_shape': line[10],
				'horizontal_visibility': line[11],
				'dew_point_temperature': line[12]
			})
		file.close()

# w = set()
# for x in total_read:
# 	for z in x['weather'].split(','):
# 		if z:
# 			w.add(re.sub(r'\s*\([^)]*\)', '', z.strip().lower()).replace(')', '').replace('(', ''))

# print(w, len(w))

features = { ## Оценка осадков от 1 до 10
	'ливень': 7,
	'сильный ливень': 9,
	'вблизи гроза': 5,
	'небольшой град и/или снежная крупа': 6,
	'сильный снег': 8, 
	'буря снег': 10,
	'слабый ливень': 4,
	'град': 9,
	'слабый снег': 3,
	'дождь': 6,
	'слабый гроза': 4,
	'вблизи ливень': 3,
	'морось': 2,
	'гроза': 8,
	'слабый морось': 1,
	'слабый дождь': 3,
	'сильный гроза': 10,
	'слабый замерзающий дождь': 8,
	'снег': 5
}

clear_weather = lambda s: re.sub(r'\s*\([^)]*\)', '', s.strip().lower()).replace(')', '').replace('(', '')

data = []

P = [float(x['sea_level_pressure']) for x in total_read]
T = [float(x['temperature']) for x in total_read]
P_mean, P_std = np.mean(P), np.std(P)
T_mean, T_std = np.mean(T), np.std(T)

W_epsilon = 0.001
W = np.log(np.array([float(x['wind_speed']) for x in total_read]) + W_epsilon)
W_mean, W_std = np.mean(W), np.std(W)

for x in total_read:
	line = {}
	## Sin/Cos Date/Time Encoding
	dt = datetime.strptime(x['date'], "%d.%m.%Y %H:%M")
	line['hour_sin'] = np.sin(2 * np.pi * dt.hour / 24)
	line['hour_cos'] = np.cos(2 * np.pi * dt.hour / 24)
	line['day_of_year_sin'] = np.sin(2 * np.pi * dt.timetuple().tm_yday / 365.25)
	line['day_of_year_cos'] = np.cos(2 * np.pi * dt.timetuple().tm_yday / 365.25)

	## Precipitation classification
	w = [features.get(clear_weather(a), 0) for a in x['weather'].split(',')]
	line['precipation'] = sum(w) / len(w)

	## Temperature, Pressure and Humidity scale
	line['pressure'] = (float(x['sea_level_pressure']) - P_mean) / P_std
	line['temperature'] = (float(x['temperature']) - T_mean) / T_std
	line['humidity'] = int(x['humidity']) / 100

	## Wind Speed normalization
	line['wind_speed'] = (np.log(float(x['wind_speed']) + W_epsilon) - W_mean) / W_std

	data.append(line)