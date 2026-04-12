package weatherHandler

import (
	"encoding/json"
	"net/http"
)

type weatherHandlerDeps struct {
	*WeatherRepo
}

func NewWeatherHandler(r *http.ServeMux, linkRepo *WeatherRepo) {
	handler := weatherHandlerDeps{linkRepo}

	r.HandleFunc("GET /weather", handler.getHandler())
	r.HandleFunc("POST /weather", handler.createHandler())
}

func (handler *weatherHandlerDeps) getHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {

		//ToDo connect with misha's service

		weatherForecast := handler.GetWeather()

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(weatherForecast)
	}
}

func (handler *weatherHandlerDeps) createHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
	}
}
