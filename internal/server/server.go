// Package server exposes the engine as an OpenAI-compatible HTTP service. It owns the transport
// (gin routing, SSE framing) and the OpenAI wire format (decode requests into engine.Request,
// encode engine.Result back into OpenAI JSON / SSE); all generation lives in internal/engine.
package server

import (
	"fmt"
	"math/rand/v2"
	"net/http"
	"time"

	"flashqwen/internal/engine"

	"github.com/gin-gonic/gin"
)

type Server struct {
	gen   *engine.Generator
	model string
}

func New(gen *engine.Generator, model string) *Server {
	return &Server{gen: gen, model: model}
}

// Run registers the routes and serves on addr (blocking).
func (s *Server) Run(addr string) error {
	gin.SetMode(gin.ReleaseMode)
	r := gin.Default()
	r.GET("/healthz", func(c *gin.Context) { c.String(http.StatusOK, "ok") })
	r.GET("/v1/models", s.listModels)
	r.POST("/v1/chat/completions", s.chatCompletions)
	r.POST("/v1/completions", s.completions)
	return r.Run(addr)
}

// newID returns a unique completion id. Random rather than an incrementing counter: there is no
// shared mutable state to race on under concurrent requests, and it matches OpenAI's opaque
// `chatcmpl-…` form. 128 random bits make collisions negligible; math/rand/v2's top-level funcs
// are concurrency-safe (and an id needs uniqueness, not cryptographic strength).
func (s *Server) newID() string {
	return fmt.Sprintf("chatcmpl-%016x%016x", rand.Uint64(), rand.Uint64())
}

func (s *Server) listModels(c *gin.Context) {
	c.JSON(http.StatusOK, ModelList{
		Object: "list",
		Data:   []Model{{ID: s.model, Object: "model", Created: time.Now().Unix(), OwnedBy: "flashqwen"}},
	})
}
