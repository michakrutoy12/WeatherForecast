package weatherHandler

import "github.com/jackc/pgx/v5/pgtype"

type GetForecastPayload struct {
	ForecastDate  pgtype.Date `json:"forecast_date"`
	ForecastHour  int         `json:"forecast_hour"`
	Temperature   float32     `json:"temperature"`
	Humidity      byte        `json:"humidity"`
	Pressure      uint16      `json:"pressure"`
	WindSpeed     float32     `json:"wind_speed"`
	Precipitation float32     `json:"precipitation"`
}
