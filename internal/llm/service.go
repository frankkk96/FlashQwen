// Package llm is the format-neutral generation core. It renders the ChatML prompt, tokenises,
// drives the token engine over gRPC, and decodes the returned id stream into text + tool calls —
// all in terms of the neutral chat.* types. Format adapters (internal/openai, future anthropic)
// sit above it and only translate their wire formats to/from llm.Request / llm.Result; the
// transport (gin) sits above them. Nothing below this package knows about HTTP or any wire format,
// and nothing above it touches enginepb.
package llm

import (
	"context"

	"flashqwen/internal/chat"
	"flashqwen/internal/engine"
	pb "flashqwen/internal/enginepb"
	"flashqwen/internal/tokenizer"
)

const defaultMaxTokens = 512

type Service struct {
	eng    *engine.Client
	cm     *chat.Model
	tok    *tokenizer.Tokenizer
	maxCtx int
}

func NewService(eng *engine.Client, cm *chat.Model, tok *tokenizer.Tokenizer, maxCtx int) *Service {
	return &Service{eng: eng, cm: cm, tok: tok, maxCtx: maxCtx}
}

// Generate runs one completion. onDelta (may be nil) is invoked for each visible text delta and
// each completed tool call as they arrive — streaming adapters forward these; blocking adapters
// pass nil and read the returned Result. Either way the returned Result holds the aggregated text,
// tool calls, finish reason, and token usage.
func (s *Service) Generate(ctx context.Context, req Request,
	onDelta func(text string, tc *chat.ToolCall)) (*Result, error) {

	g, err := s.buildRequest(req)
	if err != nil {
		return nil, err
	}

	res := &Result{}
	stream := s.cm.NewStream()
	done, err := s.eng.Generate(ctx, g, func(id int32) {
		text, tc := stream.Push(int(id))
		if text != "" {
			res.Text += text
		}
		if tc != nil {
			res.ToolCalls = append(res.ToolCalls, *tc)
		}
		if onDelta != nil && (text != "" || tc != nil) {
			onDelta(text, tc)
		}
	})
	if err != nil {
		return nil, err
	}

	res.FinishReason = done.FinishReason
	res.PromptTokens = int(done.PromptTokens)
	res.CompletionTokens = int(done.CompletionTokens)
	if len(res.ToolCalls) > 0 {
		res.FinishReason = "tool_calls"
	}
	return res, nil
}

// buildRequest renders the ChatML prompt, tokenises, clamps max_tokens to the context window, and
// attaches the stop ids — turning a neutral Request into the engine's token-level request.
func (s *Service) buildRequest(req Request) (*pb.GenerateRequest, error) {
	prompt := s.cm.Render(req.Messages, req.Tools, req.EnableThinking)
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
		maxTokens = defaultMaxTokens
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
