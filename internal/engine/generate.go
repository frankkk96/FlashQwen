package engine

import (
	"context"
	"errors"
	"fmt"

	"flashqwen/internal/chatml"
	pb "flashqwen/internal/enginepb"
	"flashqwen/internal/tokenizer"
)

// ErrPromptTooLong is returned by Generate when the prompt alone fills the context window, leaving
// no room to generate. It is a client/request error (the caller sent too much), distinct from an
// engine failure — the server maps it to HTTP 400 rather than 502.
var ErrPromptTooLong = errors.New("prompt exceeds context window")

// Generator is the text tier: it wraps a transport Client with the model-text layer (ChatML
// rendering, tokeniser, and the length policy) and turns neutral messages into completions.
type Generator struct {
	conn   *Client
	cm     *chatml.Format
	tok    *tokenizer.Tokenizer
	maxCtx int // the engine's authoritative context window (from Client.Ready), for length clamping
}

// NewGenerator builds the text tier over an already-ready transport Client. maxCtx is the engine's
// context window as reported by Client.Ready.
func NewGenerator(conn *Client, cm *chatml.Format, tok *tokenizer.Tokenizer, maxCtx int) *Generator {
	return &Generator{conn: conn, cm: cm, tok: tok, maxCtx: maxCtx}
}

// Generate runs one text completion — the high-level API for callers above this package. It renders
// the ChatML prompt, tokenises, drives the engine, and decodes the id stream into text + tool calls.
// onDelta (may be nil) is invoked for each visible text delta and completed tool call as they
// arrive — streaming callers forward these; blocking callers pass nil and read the returned Result.
// Either way the Result holds the aggregated text, tool calls, finish reason, and token usage.
func (g *Generator) Generate(ctx context.Context, req Request,
	onDelta func(text string, tc *chatml.ToolCall)) (*Result, error) {

	greq, err := g.buildRequest(req)
	if err != nil {
		return nil, err
	}

	res := &Result{}
	stream := g.cm.NewStream()
	stats, err := g.conn.stream(ctx, greq, func(id int32) {
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
	// Emit any trailing partial rune the stream held back (e.g. max_tokens hit mid-character), so
	// the aggregated text matches a whole-list decode.
	if tail := stream.Flush(); tail != "" {
		res.Text += tail
		if onDelta != nil {
			onDelta(tail, nil)
		}
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
// the prompt length. It is the only *active* length policy — callers never cap themselves (the
// engine keeps a defensive clamp of its own as a backstop, but Go always sends a resolved value):
//   - the prompt already fills the context window  -> error
//   - requested <= 0 (omitted)                      -> fill the remaining window (until eos or full)
//   - requested > the remaining window              -> clamp to the remaining window
//
// maxCtx is the engine's authoritative context window, recorded by Ready (from GetModel).
func (g *Generator) resolveMaxTokens(promptLen, requested int) (int, error) {
	room := g.maxCtx - promptLen
	if room < 1 {
		return 0, fmt.Errorf("%w: prompt is %d tokens, context window is %d", ErrPromptTooLong, promptLen, g.maxCtx)
	}
	if requested <= 0 || requested > room {
		return room, nil
	}
	return requested, nil
}

// buildRequest renders the ChatML prompt, tokenises, resolves the output budget, and attaches the
// stop ids — turning a neutral Request into the engine's token-level request.
func (g *Generator) buildRequest(req Request) (*pb.GenerateRequest, error) {
	prompt := g.cm.Render(req.Messages, req.Tools, req.EnableThinking)
	ids, err := g.tok.Encode(prompt)
	if err != nil {
		return nil, err
	}
	maxTokens, err := g.resolveMaxTokens(len(ids), req.MaxTokens)
	if err != nil {
		return nil, err
	}
	in := make([]int32, len(ids))
	for i, id := range ids {
		in[i] = int32(id)
	}

	greq := &pb.GenerateRequest{
		InputIds:     in,
		MaxTokens:    int32(maxTokens),
		TopP:         1.0,
		StopTokenIds: g.cm.StopTokenIDs(),
	}
	if req.Temperature != nil {
		greq.Temperature = float32(*req.Temperature)
	}
	if req.TopP != nil {
		greq.TopP = float32(*req.TopP)
	}
	return greq, nil
}
