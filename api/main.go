// FlashQwen OpenAI-compatible API gateway.
//
// A thin Gin server that speaks the OpenAI Chat Completions API and forwards work to the C++
// `flashqwen serve` engine over gRPC. The gateway owns only the OpenAI protocol and SSE; the
// engine owns everything model-specific (Qwen3 chat template, tokenisation, generation, tool
// calls). The gateway sends structured messages + tools and receives typed events.
//
//	flashqwen serve --model <dir> --address 127.0.0.1:50051   # start the engine (C++)
//	./flashqwen-api --engine 127.0.0.1:50051 --addr :8000     # start this gateway
package main

import (
	"context"
	"flag"
	"log"
	"time"

	"github.com/gin-gonic/gin"
)

func main() {
	addr := flag.String("addr", ":8000", "HTTP listen address")
	engineAddr := flag.String("engine", "127.0.0.1:50051", "C++ engine gRPC address")
	model := flag.String("model-name", "qwen3-8b", "fallback model id (the engine's id is used if reachable)")
	flag.Parse()

	eng, err := dialEngine(*engineAddr)
	if err != nil {
		log.Fatalf("dial engine %s: %v", *engineAddr, err)
	}
	srv := &Server{eng: eng, model: *model}
	// Adopt the engine's model id if it is already up (non-fatal if not).
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	if id, err := eng.modelID(ctx); err == nil && id != "" {
		srv.model = id
	}
	cancel()

	gin.SetMode(gin.ReleaseMode)
	r := gin.Default()
	r.GET("/v1/models", srv.listModels)
	r.POST("/v1/chat/completions", srv.chatCompletions)
	r.GET("/healthz", func(c *gin.Context) { c.String(200, "ok") })

	log.Printf("flashqwen-api on %s -> engine %s (model %q)", *addr, *engineAddr, srv.model)
	if err := r.Run(*addr); err != nil {
		log.Fatal(err)
	}
}
