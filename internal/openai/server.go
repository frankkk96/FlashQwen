// Package openai is the OpenAI-compatible HTTP layer. It renders the prompt + tokenises (chat,
// tokenizer), drives the token engine over gRPC, and turns the returned id stream back into
// OpenAI text deltas / tool calls.
package openai

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"flashqwen/internal/chat"
	"flashqwen/internal/engine"
	pb "flashqwen/internal/enginepb"
	"flashqwen/internal/tokenizer"

	"github.com/gin-gonic/gin"
)

type Server struct {
	eng    *engine.Client
	cm     *chat.Model
	tok    *tokenizer.Tokenizer
	model  string
	maxCtx int
	idSeq  int
}

func NewServer(eng *engine.Client, cm *chat.Model, tok *tokenizer.Tokenizer, model string, maxCtx int) *Server {
	return &Server{eng: eng, cm: cm, tok: tok, model: model, maxCtx: maxCtx}
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

// buildRequest maps an OpenAI chat request into the engine's token-level request: render the
// ChatML prompt, tokenise, clamp max_tokens to the context window, attach the stop ids.
func (s *Server) buildRequest(req ChatRequest) (*pb.GenerateRequest, error) {
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
	enableThinking := req.EnableThinking != nil && *req.EnableThinking

	prompt := s.cm.Render(msgs, tools, enableThinking)
	ids, err := s.tok.Encode(prompt)
	if err != nil {
		return nil, err
	}
	in := make([]int32, len(ids))
	for i, id := range ids {
		in[i] = int32(id)
	}

	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 512
	}
	if room := s.maxCtx - len(ids); room > 0 && maxTokens > room {
		maxTokens = room
	}

	g := &pb.GenerateRequest{
		InputIds:     in,
		MaxTokens:    int32(maxTokens),
		TopP:         1.0,
		StopTokenIds: s.cm.StopTokenIDs(),
	}
	if req.Temperature != nil {
		g.Temperature = float32(*req.Temperature)
	}
	if req.TopP != nil {
		g.TopP = float32(*req.TopP)
	}
	return g, nil
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
	g, err := s.buildRequest(req)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": gin.H{"message": err.Error(), "type": "tokenize_error"}})
		return
	}
	if req.Stream {
		s.streamChat(c, g)
	} else {
		s.blockingChat(c, g)
	}
}

func (s *Server) blockingChat(c *gin.Context, g *pb.GenerateRequest) {
	stream := s.cm.NewStream()
	var sb strings.Builder
	var calls []ToolCall
	done, err := s.eng.Generate(c.Request.Context(), g, func(id int32) {
		if text, tc := stream.Push(int(id)); text != "" || tc != nil {
			sb.WriteString(text)
			if tc != nil {
				calls = append(calls, ToolCall{ID: tc.ID, Type: "function",
					Function: ToolCallFunction{Name: tc.Name, Arguments: tc.ArgumentsJSON}})
			}
		}
	})
	if err != nil {
		c.JSON(http.StatusBadGateway, gin.H{"error": gin.H{"message": "engine: " + err.Error(), "type": "engine_error"}})
		return
	}
	msg := RespMessage{Role: "assistant"}
	reason := done.FinishReason
	if len(calls) > 0 {
		msg.ToolCalls = calls
		reason = "tool_calls"
		if sb.Len() > 0 {
			t := sb.String()
			msg.Content = &t
		}
	} else {
		t := sb.String()
		msg.Content = &t
	}
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
	stream := s.cm.NewStream()
	sawTool := false
	done, err := s.eng.Generate(c.Request.Context(), g, func(tid int32) {
		text, tc := stream.Push(int(tid))
		if text != "" {
			d := text
			chunk(&RespMessage{Content: &d}, nil)
		}
		if tc != nil {
			sawTool = true
			chunk(&RespMessage{ToolCalls: []ToolCall{{ID: tc.ID, Type: "function",
				Function: ToolCallFunction{Name: tc.Name, Arguments: tc.ArgumentsJSON}}}}, nil)
		}
	})
	reason := "stop"
	if err == nil && done != nil {
		reason = done.FinishReason
	}
	if sawTool {
		reason = "tool_calls"
	}
	chunk(&RespMessage{}, &reason)
	fmt.Fprint(c.Writer, "data: [DONE]\n\n")
	flusher.Flush()
}
