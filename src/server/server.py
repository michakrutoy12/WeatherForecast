from flask import Flask, jsonify
from src.predict import WeatherPredictModel, compute

app = Flask(__name__)

@app.route('/weather', methods=['GET'])
def main():
	return jsonify({'samples': {'temperature': [], 'precipation': [], 'wind': []}}) # N = Days * Sample Rate