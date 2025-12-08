package main

import (
	"log"
	"log/slog"
	"net/http"
	"os"
	"weatherForecaster/internal/cfg"

	"github.com/joho/godotenv"
)

func main() {
	err := godotenv.Load("../../../.env")
	if err != nil {
		log.Fatal("Error loading .env file", err)
	}

	config := cfg.MustLoad()
	log := setupLogger(config.Env)
	log.Info("Starting server")

	r := http.NewServeMux()
	s := http.Server{
		Addr:        config.Address,
		Handler:     r,
		IdleTimeout: config.IdleTimeout,
		ReadTimeout: config.Timeout,
	}
	log.Info(s.Addr)
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
