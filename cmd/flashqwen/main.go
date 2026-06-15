// flashqwen — the entry point. Owns the model-text layer (tokenizer, ChatML template, tool-call
// detection) and the OpenAI HTTP API; drives the C++ token engine over gRPC.
//
// Phase 2: connects to a manually-started engine (flashqwen-engine --address ...). The embedded
// engine supervisor and the chat/benchmark subcommands arrive in phase 3.
package main

import (
	"context"
	"flag"
	"log"
	"time"

	"flashqwen/internal/chat"
	"flashqwen/internal/engine"
	"flashqwen/internal/openai"
	"flashqwen/internal/tokenizer"

	"github.com/gin-gonic/gin"
)

func main() {
	addr := flag.String("addr", ":8000", "HTTP listen address")
	engineAddr := flag.String("engine", "127.0.0.1:50051", "C++ token engine gRPC address")
	modelDir := flag.String("model", "", "model directory (tokenizer.json, config.json, ...)")
	flag.Parse()
	if *modelDir == "" {
		log.Fatal("--model is required")
	}

	tok, err := tokenizer.Load(*modelDir)
	if err != nil {
		log.Fatalf("load tokenizer: %v", err)
	}
	cm := chat.Load(*modelDir, tok)

	eng, err := engine.Dial(*engineAddr)
	if err != nil {
		log.Fatalf("dial engine %s: %v", *engineAddr, err)
	}

	// Authoritative id + context window from the engine.
	modelID, maxCtx := "flashqwen", 4096
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	if info, err := eng.GetModel(ctx); err == nil {
		if info.Id != "" {
			modelID = info.Id
		}
		if info.MaxCtx > 0 {
			maxCtx = int(info.MaxCtx)
		}
	} else {
		log.Printf("warning: GetModel failed (%v); using defaults", err)
	}
	cancel()

	gin.SetMode(gin.ReleaseMode)
	r := gin.Default()
	openai.NewServer(eng, cm, tok, modelID, maxCtx).Routes(r)

	log.Printf("flashqwen on %s -> engine %s (model %q, max_ctx %d)", *addr, *engineAddr, modelID, maxCtx)
	if err := r.Run(*addr); err != nil {
		log.Fatal(err)
	}
}
