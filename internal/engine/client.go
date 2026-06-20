// Package engine is the Go-side bridge to the C++ token engine (flashqwen-engine). It is two tiers,
// one composed on top of the other, split across two files:
//
//   - client.go:   Client — the transport tier. The gRPC connection + the low-level token API
//     (Dial/Ready/GetModel/Stream); speaks only token ids, knows nothing about text.
//   - generate.go: Generator — the text tier. Wraps a Client and adds the model-text layer (ChatML
//     render + tokeniser + length policy); turns messages into a completion via Generate.
//
// enginepb (the generated gRPC stubs) is used only inside this package; nothing above it sees pb.
package engine

import (
	"context"
	"errors"
	"fmt"
	"io"
	"time"

	pb "flashqwen/internal/enginepb"

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

// Client is the transport tier: the gRPC connection plus the low-level token API. It speaks only
// token ids — the text layer (ChatML, tokeniser, length policy) lives in Generator on top of it.
type Client struct {
	conn *grpc.ClientConn
	cli  pb.EngineClient
}

// Dial connects to the engine at addr.
func Dial(addr string) (*Client, error) {
	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, err
	}
	return &Client{conn: conn, cli: pb.NewEngineClient(conn)}, nil
}

func (c *Client) Close() error { return c.conn.Close() }

// WaitReady polls GetModel until the engine answers — which, since the engine loads the model before
// binding its port, means it is loaded and serving — and returns its metadata. It fails if the engine
// process dies (aliveCheck, may be nil) or if `timeout` elapses first. Load progress is written to the
// engine's stderr.
func (c *Client) WaitReady(timeout time.Duration, aliveCheck func() error) (*ModelInfo, error) {
	deadline := time.Now().Add(timeout)
	for {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		info, err := c.GetModel(ctx)
		cancel()
		if err == nil {
			return info, nil
		}
		if aliveCheck != nil {
			if exitErr := aliveCheck(); exitErr != nil {
				return nil, exitErr
			}
		}
		if time.Now().After(deadline) {
			return nil, fmt.Errorf("engine did not become ready within %s", timeout)
		}
		time.Sleep(300 * time.Millisecond)
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
