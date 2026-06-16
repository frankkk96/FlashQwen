package engine

import (
	"context"
	"fmt"

	"flashqwen/internal/chatml"
	pb "flashqwen/internal/enginepb"
)

// Generate runs one text completion — the high-level API for callers above this package. It renders
// the ChatML prompt, tokenises, drives the engine, and decodes the id stream into text + tool calls.
// onDelta (may be nil) is invoked for each visible text delta and completed tool call as they
// arrive — streaming callers forward these; blocking callers pass nil and read the returned Result.
// Either way the Result holds the aggregated text, tool calls, finish reason, and token usage.
func (c *Client) Generate(ctx context.Context, req Request,
	onDelta func(text string, tc *chatml.ToolCall)) (*Result, error) {

	g, err := c.buildRequest(req)
	if err != nil {
		return nil, err
	}

	res := &Result{}
	stream := c.cm.NewStream()
	stats, err := c.stream(ctx, g, func(id int32) {
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

	res.FinishReason = stats.FinishReason
	res.PromptTokens = stats.PromptTokens
	res.CompletionTokens = stats.CompletionTokens
	if len(res.ToolCalls) > 0 {
		res.FinishReason = "tool_calls"
	}
	return res, nil
}

// resolveMaxTokens is the single place that decides how many tokens a request may generate, given
// the prompt length. It is the only length policy in the codebase — callers never cap themselves:
//   - the prompt already fills the context window  -> error
//   - requested <= 0 (omitted)                      -> fill the remaining window (until eos or full)
//   - requested > the remaining window              -> clamp to the remaining window
//
// maxCtx is the engine's authoritative context window, recorded via SetMaxCtx after GetModel.
func (c *Client) resolveMaxTokens(promptLen, requested int) (int, error) {
	room := c.maxCtx - promptLen
	if room < 1 {
		return 0, fmt.Errorf("prompt is %d tokens, exceeds context window %d", promptLen, c.maxCtx)
	}
	if requested <= 0 || requested > room {
		return room, nil
	}
	return requested, nil
}

// buildRequest renders the ChatML prompt, tokenises, resolves the output budget, and attaches the
// stop ids — turning a neutral Request into the engine's token-level request.
func (c *Client) buildRequest(req Request) (*pb.GenerateRequest, error) {
	prompt := c.cm.Render(req.Messages, req.Tools, req.EnableThinking)
	ids, err := c.tok.Encode(prompt)
	if err != nil {
		return nil, err
	}
	maxTokens, err := c.resolveMaxTokens(len(ids), req.MaxTokens)
	if err != nil {
		return nil, err
	}
	in := make([]int32, len(ids))
	for i, id := range ids {
		in[i] = int32(id)
	}

	g := &pb.GenerateRequest{
		InputIds:     in,
		MaxTokens:    int32(maxTokens),
		TopP:         1.0,
		StopTokenIds: c.cm.StopTokenIDs(),
	}
	if req.Temperature != nil {
		g.Temperature = float32(*req.Temperature)
	}
	if req.TopP != nil {
		g.TopP = float32(*req.TopP)
	}
	return g, nil
}
