package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	pb "flashqwen-api/pb"
	"github.com/gin-gonic/gin"
)

// Server is the OpenAI gateway: it holds the gRPC client to the engine and the model id.
type Server struct {
	eng   *engineClient
	model string
	idSeq int
}

func (s *Server) newID() string {
	s.idSeq++
	return fmt.Sprintf("chatcmpl-%d-%d", time.Now().UnixNano(), s.idSeq)
}

func (s *Server) listModels(c *gin.Context) {
	id := s.model
	if got, err := s.eng.modelID(c.Request.Context()); err == nil && got != "" {
		id = got
	}
	c.JSON(http.StatusOK, ModelList{
		Object: "list",
		Data:   []Model{{ID: id, Object: "model", Created: time.Now().Unix(), OwnedBy: "flashqwen"}},
	})
}

func pbToolCalls(tcs []*pb.ToolCall) []ToolCall {
	out := make([]ToolCall, 0, len(tcs))
	for _, tc := range tcs {
		out = append(out, ToolCall{ID: tc.Id, Type: "function",
			Function: ToolCallFunction{Name: tc.Name, Arguments: tc.ArgumentsJson}})
	}
	return out
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
	g := toGenerateRequest(req)
	if req.Stream {
		s.streamChat(c, g)
	} else {
		s.blockingChat(c, g)
	}
}

func (s *Server) blockingChat(c *gin.Context, g *pb.GenerateRequest) {
	var sb strings.Builder
	var calls []ToolCall
	done, err := s.eng.generate(c.Request.Context(), g,
		func(t string) { sb.WriteString(t) },
		func(tc *pb.ToolCall) {
			calls = append(calls, ToolCall{ID: tc.Id, Type: "function",
				Function: ToolCallFunction{Name: tc.Name, Arguments: tc.ArgumentsJson}})
		})
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": gin.H{"message": "engine: " + err.Error(), "type": "engine_error"}})
		return
	}
	msg := RespMessage{Role: "assistant"}
	if len(calls) > 0 {
		msg.ToolCalls = calls
		if sb.Len() > 0 {
			t := sb.String()
			msg.Content = &t
		}
	} else {
		t := sb.String()
		msg.Content = &t
	}
	reason := done.FinishReason
	c.JSON(http.StatusOK, ChatCompletion{
		ID: s.newID(), Object: "chat.completion", Created: time.Now().Unix(), Model: s.model,
		Choices: []Choice{{Index: 0, Message: &msg, FinishReason: &reason}},
		Usage: &Usage{PromptTokens: int(done.PromptTokens), CompletionTokens: int(done.CompletionTokens),
			TotalTokens: int(done.PromptTokens + done.CompletionTokens)},
	})
}

func (s *Server) streamChat(c *gin.Context, g *pb.GenerateRequest) {
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
	done, err := s.eng.generate(c.Request.Context(), g,
		func(t string) { d := t; chunk(&RespMessage{Content: &d}, nil) },
		func(tc *pb.ToolCall) { chunk(&RespMessage{ToolCalls: pbToolCalls([]*pb.ToolCall{tc})}, nil) })
	reason := "stop"
	if err == nil && done != nil {
		reason = done.FinishReason
	}
	chunk(&RespMessage{}, &reason)
	fmt.Fprint(c.Writer, "data: [DONE]\n\n")
	flusher.Flush()
}
