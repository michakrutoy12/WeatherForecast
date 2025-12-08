package main

import (
	"log/slog"
	"net/http"
	"os"
	"weatherForecaster/internal/cfg"

	"github.com/joho/godotenv"
)

func main() {
	godotenv.Load("../../../.env")

	cfg := cfg.MustLoad()
	log := setupLogger(cfg.Env)
	log.Info("Starting server")

	r := http.NewServeMux()
	s := http.Server{
		Addr:        cfg.Address,
		Handler:     r,
		IdleTimeout: cfg.IdleTimeout,
		ReadTimeout: cfg.Timeout,
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
