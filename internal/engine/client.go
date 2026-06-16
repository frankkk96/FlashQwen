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
	"errors"
	"fmt"
	"io"
	"time"

	"flashqwen/internal/chatml"
	pb "flashqwen/internal/enginepb"
	"flashqwen/internal/tokenizer"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// Engine request-path errors, mirroring the proto ErrorCode taxonomy. Callers classify with
// errors.Is; the wrapped message carries the engine's detailed, human-readable context so the
// server can surface it to the client verbatim. (ErrPromptTooLong, a pre-flight check, is in
// generate.go.)
var (
	ErrOverCapacity   = errors.New("engine over capacity")  // KV pool / queue full — retryable (HTTP 503)
	ErrEngineInternal = errors.New("engine internal error") // unexpected engine failure (HTTP 502)
)

// errorFor turns a structured engine Error event into a typed Go error, preserving the engine's
// detailed message verbatim.
func errorFor(e *pb.Error) error {
	switch e.GetCode() {
	case pb.ErrorCode_ERROR_CODE_OVER_CAPACITY:
		return fmt.Errorf("%w: %s", ErrOverCapacity, e.GetMessage())
	default: // INTERNAL or an unknown/future code — treat as an engine failure, keep the detail
		return fmt.Errorf("%w: %s", ErrEngineInternal, e.GetMessage())
	}
}

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

// Ready blocks until the engine answers GetModel — which arms the client's context window for
// prompt clamping — or until aliveCheck reports the engine process died, or the timeout elapses.
// It must be called once after Dial before Generate, and returns the engine's authoritative
// metadata. aliveCheck (may be nil) lets the caller fail fast when the engine exits during startup
// instead of polling a dead process until the timeout.
func (c *Client) Ready(timeout time.Duration, aliveCheck func() error) (*ModelInfo, error) {
	deadline := time.Now().Add(timeout)
	for {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		info, err := c.GetModel(ctx)
		cancel()
		if err == nil {
			c.maxCtx = info.MaxCtx // arm the length policy (the only place maxCtx is set)
			return info, nil
		}
		if aliveCheck != nil {
			if exitErr := aliveCheck(); exitErr != nil {
				return nil, exitErr
			}
		}
		if time.Now().After(deadline) {
			return nil, fmt.Errorf("engine did not become ready within %s: %w", timeout, err)
		}
		time.Sleep(500 * time.Millisecond)
	}
}

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
		case *pb.GenerateEvent_Error:
			return nil, errorFor(x.Error)
		}
	}
}
