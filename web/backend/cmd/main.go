package main

import (
	"fmt"
	"log"
	"log/slog"
	"net/http"
	"os"
	"weatherForecaster/internal/cfg"
	database "weatherForecaster/internal/db"
	"weatherForecaster/internal/services/weatherHandler"

	"github.com/joho/godotenv"
	"github.com/rs/cors"
)

func main() {
	err := godotenv.Load("../.env")
	if err != nil {
		log.Fatal("Error loading .env file", err)
	}

	config := cfg.MustLoad()
	log := setupLogger(config.Env)
	log.Info("Starting server")

	//database init
	db := database.NewDB()

	//weatherHandler
	weatherRepo := weatherHandler.NewWeatherRepo(db)
	c := cors.New(cors.Options{
		AllowedOrigins:   []string{"http://127.0.0.1:5500"}, // Замените на свой домен
		AllowedMethods:   []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowedHeaders:   []string{"Accept", "Content-Type", "Authorization"},
		AllowCredentials: true, // Разрешить куки (нельзя использовать с "*")
		Debug:            false,
	})
	r := http.NewServeMux()
	handler := c.Handler(r)
	s := http.Server{
		Addr:        config.Address,
		Handler:     handler,
		IdleTimeout: config.IdleTimeout,
		ReadTimeout: config.Timeout,
	}
	fmt.Println("http://" + s.Addr)

	weatherHandler.NewWeatherHandler(r, weatherRepo)
	// Настройка CORS

	s.ListenAndServe()
}

func setupLogger(env string) *slog.Logger {
	var log *slog.Logger

	switch env {
	case "local":
		log = slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelDebug}))
	case "dev":
		log = slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelDebug}))
	case "prod":
		log = slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))
	}

	return log
}
