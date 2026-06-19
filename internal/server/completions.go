package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"time"

	"flashqwen/internal/chatml"
	"flashqwen/internal/engine"

	"github.com/gin-gonic/gin"
)

// classifyError maps a Generate error to an OpenAI error type + HTTP status. The error's message
// (which carries the engine's detailed context) is surfaced to the client separately; this only
// decides the category.
func classifyError(err error) (status int, typ string) {
	switch {
	case errors.Is(err, engine.ErrPromptTooLong): // prompt overflows the context window — client error
		return http.StatusBadRequest, "invalid_request_error"
	case errors.Is(err, engine.ErrOverCapacity): // KV pool / queue full — retryable
		return http.StatusServiceUnavailable, "overloaded_error"
	default: // ErrEngineInternal, transport failures, anything else
		return http.StatusBadGateway, "engine_error"
	}
}

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
		IgnoreEOS:      req.IgnoreEOS,
	}
}

// decodeCompletion maps an OpenAI /v1/completions request into the neutral engine.Request. prompt is
// either a string (tokenised as-is) or an array of token ids.
func decodeCompletion(req CompletionRequest) (engine.Request, error) {
	er := engine.Request{
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		IgnoreEOS:   req.IgnoreEOS,
	}
	var s string
	if json.Unmarshal(req.Prompt, &s) == nil {
		er.RawPrompt = s
		return er, nil
	}
	var ids []int
	if json.Unmarshal(req.Prompt, &ids) == nil {
		er.InputIDs = ids
		return er, nil
	}
	return er, fmt.Errorf("prompt must be a string or an array of token ids")
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
		includeUsage := req.StreamOptions != nil && req.StreamOptions.IncludeUsage
		s.streamChat(c, decode(req), includeUsage)
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
	res, err := s.gen.Generate(c.Request.Context(), req, nil)
	if err != nil {
		status, typ := classifyError(err)
		c.JSON(status, gin.H{"error": gin.H{"message": err.Error(), "type": typ}})
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

func (s *Server) streamChat(c *gin.Context, req engine.Request, includeUsage bool) {
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
	res, err := s.gen.Generate(c.Request.Context(), req, func(text string, tc *chatml.ToolCall) {
		if text != "" {
			d := text
			chunk(&RespMessage{Content: &d}, nil)
		}
		if tc != nil {
			chunk(&RespMessage{ToolCalls: toolCalls([]chatml.ToolCall{*tc})}, nil)
		}
	})
	if err != nil {
		// Client disconnected mid-stream: not a server error, just stop (the connection is gone).
		if c.Request.Context().Err() != nil {
			return
		}
		// The 200 + SSE headers are already flushed, so we can't switch to an HTTP error status;
		// surface the failure as an SSE error event rather than a misleading finish_reason.
		_, typ := classifyError(err)
		b, _ := json.Marshal(gin.H{"error": gin.H{"message": err.Error(), "type": typ}})
		fmt.Fprintf(c.Writer, "data: %s\n\n", b)
		fmt.Fprint(c.Writer, "data: [DONE]\n\n")
		flusher.Flush()
		return
	}
	reason := res.FinishReason
	chunk(&RespMessage{}, &reason)
	if includeUsage {
		// Per the OpenAI stream_options contract: a final chunk with an empty choices array and the
		// token usage. Benchmarks (and metering clients) rely on this for an exact output-token count.
		usage := ChatCompletion{ID: id, Object: "chat.completion.chunk", Created: created, Model: s.model,
			Choices: []Choice{}, Usage: &Usage{PromptTokens: res.PromptTokens, CompletionTokens: res.CompletionTokens,
				TotalTokens: res.PromptTokens + res.CompletionTokens}}
		b, _ := json.Marshal(usage)
		fmt.Fprintf(c.Writer, "data: %s\n\n", b)
	}
	fmt.Fprint(c.Writer, "data: [DONE]\n\n")
	flusher.Flush()
}

// ---- /v1/completions (raw text completion) ----------------------------------------------

func (s *Server) completions(c *gin.Context) {
	var req CompletionRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": gin.H{"message": err.Error(), "type": "invalid_request_error"}})
		return
	}
	if len(req.Prompt) == 0 {
		c.JSON(http.StatusBadRequest, gin.H{"error": gin.H{"message": "prompt is required", "type": "invalid_request_error"}})
		return
	}
	er, err := decodeCompletion(req)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": gin.H{"message": err.Error(), "type": "invalid_request_error"}})
		return
	}
	if req.Stream {
		includeUsage := req.StreamOptions != nil && req.StreamOptions.IncludeUsage
		s.streamCompletion(c, er, includeUsage)
	} else {
		s.blockingCompletion(c, er)
	}
}

func (s *Server) blockingCompletion(c *gin.Context, req engine.Request) {
	res, err := s.gen.Complete(c.Request.Context(), req, nil)
	if err != nil {
		status, typ := classifyError(err)
		c.JSON(status, gin.H{"error": gin.H{"message": err.Error(), "type": typ}})
		return
	}
	reason := res.FinishReason
	c.JSON(http.StatusOK, Completion{
		ID: s.newID(), Object: "text_completion", Created: time.Now().Unix(), Model: s.model,
		Choices: []CompletionChoice{{Index: 0, Text: res.Text, FinishReason: &reason}},
		Usage: &Usage{PromptTokens: res.PromptTokens, CompletionTokens: res.CompletionTokens,
			TotalTokens: res.PromptTokens + res.CompletionTokens},
	})
}

func (s *Server) streamCompletion(c *gin.Context, req engine.Request, includeUsage bool) {
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
	send := func(cc Completion) {
		b, _ := json.Marshal(cc)
		fmt.Fprintf(c.Writer, "data: %s\n\n", b)
		flusher.Flush()
	}
	textChunk := func(text string, reason *string) {
		send(Completion{ID: id, Object: "text_completion", Created: created, Model: s.model,
			Choices: []CompletionChoice{{Index: 0, Text: text, FinishReason: reason}}})
	}

	res, err := s.gen.Complete(c.Request.Context(), req, func(text string) {
		if text != "" {
			textChunk(text, nil)
		}
	})
	if err != nil {
		// Client disconnected mid-stream: not a server error, just stop (the connection is gone).
		if c.Request.Context().Err() != nil {
			return
		}
		_, typ := classifyError(err)
		b, _ := json.Marshal(gin.H{"error": gin.H{"message": err.Error(), "type": typ}})
		fmt.Fprintf(c.Writer, "data: %s\n\n", b)
		fmt.Fprint(c.Writer, "data: [DONE]\n\n")
		flusher.Flush()
		return
	}
	reason := res.FinishReason
	textChunk("", &reason)
	if includeUsage {
		// Final chunk: empty choices + token usage (the exact output-token count benchmarks rely on).
		send(Completion{ID: id, Object: "text_completion", Created: created, Model: s.model,
			Choices: []CompletionChoice{}, Usage: &Usage{PromptTokens: res.PromptTokens,
				CompletionTokens: res.CompletionTokens, TotalTokens: res.PromptTokens + res.CompletionTokens}})
	}
	fmt.Fprint(c.Writer, "data: [DONE]\n\n")
	flusher.Flush()
}
