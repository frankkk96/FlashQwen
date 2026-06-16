package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"flashqwen/internal/chatml"
	"flashqwen/internal/engine"

	"github.com/gin-gonic/gin"
)

// decode maps an OpenAI chat request into the neutral engine.Request.
func decode(req ChatRequest) engine.Request {
	msgs := make([]chatml.Message, 0, len(req.Messages))
	for _, m := range req.Messages {
		cm := chatml.Message{Role: m.Role, Content: m.Text(), ToolCallID: m.ToolCallID}
		for _, tc := range m.ToolCalls {
			cm.ToolCalls = append(cm.ToolCalls, chatml.ToolCall{
				ID: tc.ID, Name: tc.Function.Name, ArgumentsJSON: tc.Function.Arguments})
		}
		msgs = append(msgs, cm)
	}
	tools := make([]chatml.ToolDef, 0, len(req.Tools))
	for _, t := range req.Tools {
		tools = append(tools, chatml.ToolDef{
			Name: t.Function.Name, Description: t.Function.Description,
			ParametersJSON: string(t.Function.Parameters)})
	}
	return engine.Request{
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

func toolCalls(tcs []chatml.ToolCall) []ToolCall {
	out := make([]ToolCall, len(tcs))
	for i, tc := range tcs {
		out[i] = ToolCall{ID: tc.ID, Type: "function",
			Function: ToolCallFunction{Name: tc.Name, Arguments: tc.ArgumentsJSON}}
	}
	return out
}

func (s *Server) blockingChat(c *gin.Context, req engine.Request) {
	res, err := s.eng.Generate(c.Request.Context(), req, nil)
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

func (s *Server) streamChat(c *gin.Context, req engine.Request) {
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
	res, err := s.eng.Generate(c.Request.Context(), req, func(text string, tc *chatml.ToolCall) {
		if text != "" {
			d := text
			chunk(&RespMessage{Content: &d}, nil)
		}
		if tc != nil {
			chunk(&RespMessage{ToolCalls: toolCalls([]chatml.ToolCall{*tc})}, nil)
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
