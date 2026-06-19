package engine

import (
	"context"
	"errors"
	"fmt"
	"unicode/utf8"

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

// buildRequest resolves the prompt token ids (pre-tokenised InputIDs, a raw prompt, or a rendered
// ChatML template — in that order), resolves the output budget, and attaches the stop ids — turning
// a neutral Request into the engine's token-level request. IgnoreEOS drops the stop ids so the
// engine runs to MaxTokens (used by benchmarks to force a fixed output length).
func (g *Generator) buildRequest(req Request) (*pb.GenerateRequest, error) {
	var ids []int
	switch {
	case len(req.InputIDs) > 0:
		ids = req.InputIDs
	case req.RawPrompt != "":
		var err error
		if ids, err = g.tok.Encode(req.RawPrompt); err != nil {
			return nil, err
		}
	default:
		var err error
		if ids, err = g.tok.Encode(g.cm.Render(req.Messages, req.Tools, req.EnableThinking)); err != nil {
			return nil, err
		}
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
		InputIds:  in,
		MaxTokens: int32(maxTokens),
		TopP:      1.0,
	}
	if !req.IgnoreEOS {
		greq.StopTokenIds = g.cm.StopTokenIDs()
	}
	if req.Temperature != nil {
		greq.Temperature = float32(*req.Temperature)
	}
	if req.TopP != nil {
		greq.TopP = float32(*req.TopP)
	}
	return greq, nil
}

// Complete runs a raw text completion (no ChatML template, no tool-call parsing): it builds the
// request from RawPrompt/InputIDs, drives the engine, and decodes the id stream straight to text.
// onDelta (may be nil) receives each visible text delta as it arrives. Used by /v1/completions.
func (g *Generator) Complete(ctx context.Context, req Request, onDelta func(text string)) (*Result, error) {
	greq, err := g.buildRequest(req)
	if err != nil {
		return nil, err
	}
	res := &Result{}
	var pending []byte // decoded bytes not yet emitted (the tail may be a half-finished rune)
	stats, err := g.conn.stream(ctx, greq, func(id int32) {
		pending = append(pending, g.tok.Decode([]int{int(id)})...)
		if text := takeRunes(&pending); text != "" {
			res.Text += text
			if onDelta != nil {
				onDelta(text)
			}
		}
	})
	if err != nil {
		return nil, err
	}
	if len(pending) > 0 { // trailing partial rune (e.g. max_tokens landed mid-character)
		tail := string(pending)
		res.Text += tail
		if onDelta != nil {
			onDelta(tail)
		}
	}
	res.FinishReason = stats.FinishReason
	res.PromptTokens = stats.PromptTokens
	res.CompletionTokens = stats.CompletionTokens
	return res, nil
}

// takeRunes drains the complete-UTF-8-rune prefix of *buf, retaining any trailing partial multibyte
// sequence (a rune can straddle two tokens). Mirrors chatml.Stream's decode buffering, for the raw
// completion path which has no ChatML stream.
func takeRunes(buf *[]byte) string {
	b := *buf
	i := 0
	for i < len(b) {
		if b[i] < utf8.RuneSelf {
			i++
			continue
		}
		if !utf8.FullRune(b[i:]) {
			break
		}
		_, size := utf8.DecodeRune(b[i:])
		i += size
	}
	if i == 0 {
		return ""
	}
	out := string(b[:i])
	n := copy(b, b[i:])
	*buf = b[:n]
	return out
}
