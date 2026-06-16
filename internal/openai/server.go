// Package openai is the OpenAI-compatible HTTP adapter: it decodes OpenAI chat requests into the
// neutral llm.Request, delegates generation to llm.Service, and encodes the result back into
// OpenAI JSON / SSE. It owns only the OpenAI wire format and its transport framing; all generation
// (prompt rendering, tokenisation, the token engine, stream decoding) lives in internal/llm.
package openai

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"flashqwen/internal/chat"
	"flashqwen/internal/llm"

	"github.com/gin-gonic/gin"
)

type Server struct {
	svc   *llm.Service
	model string
	idSeq int
}

func NewServer(svc *llm.Service, model string) *Server {
	return &Server{svc: svc, model: model}
}

// Routes registers the OpenAI endpoints on a gin engine.
func (s *Server) Routes(r *gin.Engine) {
	r.GET("/v1/models", s.listModels)
	r.POST("/v1/chat/completions", s.chatCompletions)
	r.GET("/healthz", func(c *gin.Context) { c.String(200, "ok") })
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

// decode maps an OpenAI chat request into the neutral llm.Request.
func decode(req ChatRequest) llm.Request {
	msgs := make([]chat.Message, 0, len(req.Messages))
	for _, m := range req.Messages {
		cm := chat.Message{Role: m.Role, Content: m.Text(), ToolCallID: m.ToolCallID}
		for _, tc := range m.ToolCalls {
			cm.ToolCalls = append(cm.ToolCalls, chat.ToolCall{
				ID: tc.ID, Name: tc.Function.Name, ArgumentsJSON: tc.Function.Arguments})
		}
		msgs = append(msgs, cm)
	}
	tools := make([]chat.ToolDef, 0, len(req.Tools))
	for _, t := range req.Tools {
		tools = append(tools, chat.ToolDef{
			Name: t.Function.Name, Description: t.Function.Description,
			ParametersJSON: string(t.Function.Parameters)})
	}
	return llm.Request{
		Messages:       msgs,
		Tools:          tools,
		EnableThinking: req.EnableThinking != nil && *req.EnableThinking,
		MaxTokens:      req.MaxTokens,
		Temperature:    req.Temperature,
		TopP:           req.TopP,
	}
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
	if req.Stream {
		s.streamChat(c, decode(req))
	} else {
		s.blockingChat(c, decode(req))
	}
}

func toolCalls(tcs []chat.ToolCall) []ToolCall {
	out := make([]ToolCall, len(tcs))
	for i, tc := range tcs {
		out[i] = ToolCall{ID: tc.ID, Type: "function",
			Function: ToolCallFunction{Name: tc.Name, Arguments: tc.ArgumentsJSON}}
	}
	return out
}

func (s *Server) blockingChat(c *gin.Context, req llm.Request) {
	res, err := s.svc.Generate(c.Request.Context(), req, nil)
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": gin.H{"message": "engine: " + err.Error(), "type": "engine_error"}})
		return
	}
	msg := RespMessage{Role: "assistant"}
	if len(res.ToolCalls) > 0 {
		msg.ToolCalls = toolCalls(res.ToolCalls)
		if res.Text != "" {
			t := res.Text
			msg.Content = &t
		}
	} else {
		t := res.Text
		msg.Content = &t
	}
	reason := res.FinishReason
	c.JSON(http.StatusOK, ChatCompletion{
		ID: s.newID(), Object: "chat.completion", Created: time.Now().Unix(), Model: s.model,
		Choices: []Choice{{Index: 0, Message: &msg, FinishReason: &reason}},
		Usage: &Usage{PromptTokens: res.PromptTokens, CompletionTokens: res.CompletionTokens,
			TotalTokens: res.PromptTokens + res.CompletionTokens},
	})
}

func (s *Server) streamChat(c *gin.Context, req llm.Request) {
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
	res, err := s.svc.Generate(c.Request.Context(), req, func(text string, tc *chat.ToolCall) {
		if text != "" {
			d := text
			chunk(&RespMessage{Content: &d}, nil)
		}
		if tc != nil {
			chunk(&RespMessage{ToolCalls: toolCalls([]chat.ToolCall{*tc})}, nil)
		}
	})
	reason := "stop"
	if err == nil && res != nil {
		reason = res.FinishReason
	}
	chunk(&RespMessage{}, &reason)
	fmt.Fprint(c.Writer, "data: [DONE]\n\n")
	flusher.Flush()
}
