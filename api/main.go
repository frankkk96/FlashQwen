// FlashQwen OpenAI-compatible API gateway.
//
// A thin Gin server that speaks the OpenAI Chat Completions API and forwards work to the C++
// `flashqwen serve` process over a Unix socket. The gateway owns the OpenAI protocol, the Qwen3
// chat template, tool-call rendering/parsing, and SSE streaming; the engine owns tokenisation and
// continuous-batched generation.
//
//	flashqwen serve --model <dir> --socket /tmp/flashqwen.sock     # start the engine (C++)
//	./flashqwen-api --socket /tmp/flashqwen.sock --addr :8000      # start this gateway
package main

import (
	"flag"
	"log"

	"github.com/gin-gonic/gin"
)

func main() {
	addr := flag.String("addr", ":8000", "HTTP listen address")
	socket := flag.String("socket", "/tmp/flashqwen.sock", "C++ engine Unix-socket path")
	model := flag.String("model-name", "qwen3-8b", "model id reported by the API")
	flag.Parse()

	gin.SetMode(gin.ReleaseMode)
	srv := &Server{socket: *socket, model: *model}

	r := gin.Default()
	r.GET("/v1/models", srv.listModels)
	r.POST("/v1/chat/completions", srv.chatCompletions)
	r.GET("/healthz", func(c *gin.Context) { c.String(200, "ok") })

	log.Printf("flashqwen-api on %s -> engine %s (model %q)", *addr, *socket, *model)
	if err := r.Run(*addr); err != nil {
		log.Fatal(err)
	}
}
