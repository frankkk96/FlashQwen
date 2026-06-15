package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
)

// Server holds the gateway config: where the C++ engine socket is and the model id to report.
type Server struct {
	socket string
	model  string
	idSeq  int
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

// buildEngineReq maps an OpenAI request to the engine protocol (defaults: greedy, no nucleus cut).
func (s *Server) buildEngineReq(req ChatRequest) engineReq {
	er := engineReq{Prompt: renderPrompt(req), MaxTokens: req.MaxTokens, Temperature: 0, TopP: 1.0}
	if er.MaxTokens <= 0 {
		er.MaxTokens = 512
	}
	if req.Temperature != nil {
		er.Temperature = *req.Temperature
	}
	if req.TopP != nil {
		er.TopP = *req.TopP
	}
	return er
}

func (s *Server) chatCompletions(c *gin.Context) {
	var req ChatRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": gin.H{"message": err.Error(), "type": "invalid_request_error"}})
		return
	}
	if len(req.Messages) == 0 {
		c.JSON(http.StatusBadRequest, gin.H{"error": gin.H{"message": "messages is required", "type": "invalid_request_error"}})
		return
	}
	er := s.buildEngineReq(req)
	if req.Stream {
		s.streamChat(c, req, er)
	} else {
		s.blockingChat(c, req, er)
	}
}

func (s *Server) blockingChat(c *gin.Context, req ChatRequest, er engineReq) {
	var sb strings.Builder
	fin, err := generate(s.socket, er, func(d string) { sb.WriteString(d) })
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": gin.H{"message": "engine: " + err.Error(), "type": "engine_error"}})
		return
	}
	calls, clean := extractToolCalls(sb.String())

	msg := RespMessage{Role: "assistant"}
	reason := fin.FinishReason
	if len(calls) > 0 {
		msg.ToolCalls = calls
		reason = "tool_calls"
		if clean != "" {
			msg.Content = &clean
		}
	} else {
		full := sb.String()
		msg.Content = &full
	}
	c.JSON(http.StatusOK, ChatCompletion{
		ID: s.newID(), Object: "chat.completion", Created: time.Now().Unix(), Model: s.model,
		Choices: []Choice{{Index: 0, Message: &msg, FinishReason: &reason}},
		Usage: &Usage{PromptTokens: fin.PromptTokens, CompletionTokens: fin.CompletionTokens,
			TotalTokens: fin.PromptTokens + fin.CompletionTokens},
	})
}

func (s *Server) streamChat(c *gin.Context, req ChatRequest, er engineReq) {
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	flusher, ok := c.Writer.(http.Flusher)
	if !ok {
		c.JSON(http.StatusInternalServerError, gin.H{"error": gin.H{"message": "streaming unsupported"}})
		return
	}
	id := s.newID()
	created := time.Now().Unix()

	chunk := func(delta *RespMessage, reason *string) {
		cc := ChatCompletion{ID: id, Object: "chat.completion.chunk", Created: created, Model: s.model,
			Choices: []Choice{{Index: 0, Delta: delta, FinishReason: reason}}}
		b, _ := json.Marshal(cc)
		fmt.Fprintf(c.Writer, "data: %s\n\n", b)
		flusher.Flush()
	}

	chunk(&RespMessage{Role: "assistant"}, nil) // opening chunk carries the role

	if len(req.Tools) == 0 {
		// True token-by-token streaming when no tool parsing is needed.
		fin, err := generate(s.socket, er, func(d string) { chunk(&RespMessage{Content: &d}, nil) })
		reason := fin.FinishReason
		if err != nil {
			reason = "stop"
		}
		chunk(&RespMessage{}, &reason)
	} else {
		// Tool calls need the whole completion, so buffer then emit once.
		var sb strings.Builder
		fin, _ := generate(s.socket, er, func(d string) { sb.WriteString(d) })
		calls, clean := extractToolCalls(sb.String())
		reason := fin.FinishReason
		if len(calls) > 0 {
			reason = "tool_calls"
			d := &RespMessage{ToolCalls: calls}
			if clean != "" {
				d.Content = &clean
			}
			chunk(d, nil)
		} else {
			full := sb.String()
			chunk(&RespMessage{Content: &full}, nil)
		}
		chunk(&RespMessage{}, &reason)
	}
	fmt.Fprint(c.Writer, "data: [DONE]\n\n")
	flusher.Flush()
}
