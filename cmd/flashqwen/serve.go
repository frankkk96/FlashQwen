package main

import (
	"flag"
	"log"
	"os"
	"os/signal"
	"syscall"

	"flashqwen/internal/openai"

	"github.com/gin-gonic/gin"
)

func runServe(args []string) {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	model := fs.String("model", "", "model directory (required)")
	addr := fs.String("addr", ":8000", "HTTP listen address")
	slots := fs.Int("slots", 16, "max concurrent sequences")
	maxCtx := fs.Int("max-ctx", 4096, "KV / context length")
	fs.Parse(args)
	if *model == "" {
		log.Fatal("serve: --model is required")
	}

	s, err := open(*model, *slots, *maxCtx)
	if err != nil {
		log.Fatalf("serve: %v", err)
	}
	// Tear the engine down on Ctrl-C / SIGTERM (deferred funcs don't run on signal exit).
	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	go func() { <-sig; s.stop(); os.Exit(0) }()

	gin.SetMode(gin.ReleaseMode)
	r := gin.Default()
	openai.NewServer(s.eng, s.cm, s.tok, s.info.Id, s.maxCtx()).Routes(r)

	log.Printf("flashqwen serve on %s (model %q, max_ctx %d)", *addr, s.info.Id, s.maxCtx())
	if err := r.Run(*addr); err != nil {
		s.stop()
		log.Fatal(err)
	}
}
