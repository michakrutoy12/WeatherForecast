package database

import (
	"context"
	"log"
	"os"

	"github.com/jackc/pgx/v5"
)

type DB struct {
	*pgx.Conn
}

func NewDB() *DB {
	conn, err := pgx.Connect(context.Background(), os.Getenv("DATABASE_URL"))
	if err != nil {
		log.Fatalf("ERR_CONNECT_TO_DB: %s", err)
	}
	return &DB{conn}
}
