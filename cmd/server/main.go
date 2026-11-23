package main

import (
	"fmt"
	"os"
	"time"

	"etsi-rag/internal/config"
	"etsi-rag/internal/server"

	"github.com/rs/zerolog"
)

func main() {
	logger := zerolog.New(zerolog.ConsoleWriter{Out: os.Stderr, TimeFormat: time.RFC3339}).
		With().
		Timestamp().
		Logger()

	if err := run(logger); err != nil {
		logger.Fatal().Err(err).Msg("server error")
	}
}

func run(logger zerolog.Logger) error {
	configDir := "config"
	cfg, err := config.Load(configDir)
	if err != nil {
		return fmt.Errorf("loading config: %w", err)
	}

	srv, err := server.New(cfg, config.Path(configDir), logger)
	if err != nil {
		return err
	}
	return srv.Run()
}
