// Package engine is the Go-side bridge to the C++ token engine (flashqwen-engine) plus the request
// processing on top of it. The Client holds the gRPC connection together with the model-text layer
// (chat template + tokenizer) and exposes two tiers:
//
//   - Stream:   low-level token API (prompt ids in, sampled ids out) for the throughput benchmark.
//   - Generate: high-level text API (render ChatML -> tokenise -> drive the engine -> decode the id
//     stream into text + tool calls), returning a neutral Result.
//
// enginepb (the generated gRPC stubs) is used only here; nothing above this package sees it.
package engine

import (
	"context"
	"fmt"
	"io"

	"flashqwen/internal/chatml"
	pb "flashqwen/internal/enginepb"
	"flashqwen/internal/tokenizer"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

const defaultMaxTokens = 512

type Client struct {
	conn   *grpc.ClientConn
	cli    pb.EngineClient
	cm     *chatml.Format
	tok    *tokenizer.Tokenizer
	maxCtx int
}

// Dial connects to the engine at addr, binding the model-text layer used by Generate.
func Dial(addr string, cm *chatml.Format, tok *tokenizer.Tokenizer) (*Client, error) {
	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, err
	}
	return &Client{conn: conn, cli: pb.NewEngineClient(conn), cm: cm, tok: tok}, nil
}

func (c *Client) Close() error { return c.conn.Close() }

// SetMaxCtx records the engine's authoritative context window (from GetModel) for prompt clamping.
func (c *Client) SetMaxCtx(n int) { c.maxCtx = n }

// GetModel returns the engine's metadata.
func (c *Client) GetModel(ctx context.Context) (*ModelInfo, error) {
	mi, err := c.cli.GetModel(ctx, &pb.ModelRequest{})
	if err != nil {
		return nil, err
	}
	return &ModelInfo{ID: mi.Id, MaxCtx: int(mi.MaxCtx), VocabSize: int(mi.VocabSize)}, nil
}

// Stream is the low-level token API: it sends the given prompt ids and invokes onToken for each
// sampled id, returning the terminal Stats. temperature<=0 means greedy; topP=1 disables nucleus;
// stopIDs (may be nil) stop generation as soon as one is sampled. Used by the synthetic-id
// benchmark; Generate is the text-level path.
func (c *Client) Stream(ctx context.Context, ids []int32, maxTokens int,
	temperature, topP float32, stopIDs []int32, onToken func(int32)) (*Stats, error) {
	return c.stream(ctx, &pb.GenerateRequest{
		InputIds:     ids,
		MaxTokens:    int32(maxTokens),
		Temperature:  temperature,
		TopP:         topP,
		StopTokenIds: stopIDs,
	}, onToken)
}

func (c *Client) stream(ctx context.Context, g *pb.GenerateRequest, onToken func(int32)) (*Stats, error) {
	st, err := c.cli.Generate(ctx, g)
	if err != nil {
		return nil, err
	}
	for {
		ev, err := st.Recv()
		if err == io.EOF {
			return &Stats{FinishReason: "stop"}, nil
		}
		if err != nil {
			return nil, err
		}
		switch x := ev.Event.(type) {
		case *pb.GenerateEvent_TokenId:
			if onToken != nil {
				onToken(x.TokenId)
			}
		case *pb.GenerateEvent_Done:
			return &Stats{FinishReason: x.Done.FinishReason,
				PromptTokens: int(x.Done.PromptTokens), CompletionTokens: int(x.Done.CompletionTokens)}, nil
		}
	}
}

// Generate runs one text completion. onDelta (may be nil) is invoked for each visible text delta
// and completed tool call as they arrive — streaming callers forward these; blocking callers pass
// nil and read the returned Result. Either way the Result holds the aggregated text, tool calls,
// finish reason, and token usage.
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

// buildRequest renders the ChatML prompt, tokenises, clamps max_tokens to the context window, and
// attaches the stop ids — turning a neutral Request into the engine's token-level request. It
// errors if the prompt alone fills the context window.
func (c *Client) buildRequest(req Request) (*pb.GenerateRequest, error) {
	prompt := c.cm.Render(req.Messages, req.Tools, req.EnableThinking)
	ids, err := c.tok.Encode(prompt)
	if err != nil {
		return nil, err
	}
	if c.maxCtx > 0 && len(ids) >= c.maxCtx {
		return nil, fmt.Errorf("prompt is %d tokens, exceeds context window %d", len(ids), c.maxCtx)
	}
	in := make([]int32, len(ids))
	for i, id := range ids {
		in[i] = int32(id)
	}

	maxTokens := req.MaxTokens
	if maxTokens <= 0 {
		maxTokens = defaultMaxTokens
	}
	if room := c.maxCtx - len(ids); room > 0 && maxTokens > room {
		maxTokens = room
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
