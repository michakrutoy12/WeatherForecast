package weatherHandler

import (
	"context"
	database "weatherForecaster/internal/db"
)

type WeatherRepo struct {
	*database.DB
}

func NewWeatherRepo(db *database.DB) *WeatherRepo {
	return &WeatherRepo{
		db,
	}
}
func (db *WeatherRepo) GetWeather() [3]GetForecastPayload {
	var A [3]GetForecastPayload
	for i := 0; i <= 2; i++ {
		db.QueryRow(context.Background(), "SELECT forecast_date, forecast_hour, temperature, humidity, pressure, wind_speed, precipitation FROM public.weather_forecast where id=$1", i+9).Scan(&A[i].ForecastDate, &A[i].ForecastHour, &A[i].Temperature, &A[i].Humidity, &A[i].Pressure, &A[i].WindSpeed, &A[i].Precipitation)
	}
	return A
}
