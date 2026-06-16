// Package engine is the Go-side bridge to the C++ token engine (flashqwen-engine) plus the request
// processing on top of it. A single Client holds the gRPC connection together with the model-text
// layer (ChatML format + tokenizer) and exposes two tiers, split across two files:
//
//   - client.go:   low-level token API — Dial/GetModel/Stream — talking to the backend in token ids.
//   - generate.go: high-level text API — Generate — render ChatML, tokenise, drive the engine, and
//     decode the id stream into text + tool calls.
//
// enginepb (the generated gRPC stubs) is used only inside this package; nothing above it sees pb.
package engine

import (
	"context"
	"io"

	"flashqwen/internal/chatml"
	pb "flashqwen/internal/enginepb"
	"flashqwen/internal/tokenizer"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

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

// stream is the shared gRPC driver: it opens the Generate RPC and pumps token-id events into
// onToken until the terminal Done. Both the low-level Stream and high-level Generate go through it.
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
