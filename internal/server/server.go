// Package server exposes the engine as an OpenAI-compatible HTTP service. It owns the transport
// (gin routing, SSE framing) and the OpenAI wire format (decode requests into engine.Request,
// encode engine.Result back into OpenAI JSON / SSE); all generation lives in internal/engine.
package server

import (
	"fmt"
	"net/http"
	"time"

	"flashqwen/internal/engine"

	"github.com/gin-gonic/gin"
)

type Server struct {
	eng   *engine.Client
	model string
	idSeq int
}

func New(eng *engine.Client, model string) *Server {
	return &Server{eng: eng, model: model}
}

// Run registers the routes and serves on addr (blocking).
func (s *Server) Run(addr string) error {
	gin.SetMode(gin.ReleaseMode)
	r := gin.Default()
	r.GET("/healthz", func(c *gin.Context) { c.String(http.StatusOK, "ok") })
	r.GET("/v1/models", s.listModels)
	r.POST("/v1/chat/completions", s.chatCompletions)
	return r.Run(addr)
}

func (s *Server) newID() string {
	s.idSeq++
	return fmt.Sprintf("chatcmpl-%d-%d", time.Now().UnixNano(), s.idSeq)
}

func (s *Server) listModels(c *gin.Context) {
	c.JSON(http.StatusOK, ModelList{
		Object: "list",
		Data:   []Model{{ID: s.model, Object: "model", Created: time.Now().Unix(), OwnedBy: "flashqwen"}},
	})
}
