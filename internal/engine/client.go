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

// StatusState is the engine's startup lifecycle, mirroring the proto EngineState.
type StatusState int

const (
	StateLoading StatusState = iota // binding done; weights / KV pool loading
	StateReady                      // serving
	StateFailed                     // load failed (terminal); Message has the cause
)

// Status is a snapshot of the engine's startup progress (one GetStatus reply).
type Status struct {
	State       StatusState
	Phase       string // e.g. "loading weights" / "allocating kv pool"
	Done, Total int    // progress within Phase (e.g. layers); 0 when not countable
	Message     string // failure cause when State == StateFailed
}

func stateFromProto(s pb.EngineState) StatusState {
	switch s {
	case pb.EngineState_ENGINE_STATE_READY:
		return StateReady
	case pb.EngineState_ENGINE_STATE_FAILED:
		return StateFailed
	default:
		return StateLoading
	}
}

// GetStatus returns the engine's current startup status. It is answerable from the moment the engine
// binds its port — before the model is loaded — so callers can show progress and detect failure.
func (c *Client) GetStatus(ctx context.Context) (*Status, error) {
	s, err := c.cli.GetStatus(ctx, &pb.StatusRequest{})
	if err != nil {
		return nil, err
	}
	return &Status{State: stateFromProto(s.GetState()), Phase: s.GetPhase(),
		Done: int(s.GetDone()), Total: int(s.GetTotal()), Message: s.GetMessage()}, nil
}

// WaitReady polls GetStatus until the engine is ready (returning its authoritative metadata via
// GetModel), the load fails, or it stalls. aliveCheck (may be nil) fails fast if the engine process
// dies. onStatus (may be nil) is invoked on every poll so callers can render a progress bar.
//
// The watchdog is stall-based, not an absolute deadline: as long as the load keeps making progress
// (phase or done/total advancing) it waits indefinitely, so a slow load of a large model is never
// killed prematurely; only genuinely stuck progress (no change for stallTimeout, process still
// alive) gives up.
func (c *Client) WaitReady(stallTimeout time.Duration, aliveCheck func() error, onStatus func(Status)) (*ModelInfo, error) {
	var last Status
	haveStatus := false
	lastChange := time.Now()
	for {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		st, err := c.GetStatus(ctx)
		cancel()
		if err == nil {
			if !haveStatus || st.State != last.State || st.Phase != last.Phase ||
				st.Done != last.Done || st.Total != last.Total {
				lastChange = time.Now()
			}
			last, haveStatus = *st, true
			if onStatus != nil {
				onStatus(*st)
			}
			switch st.State {
			case StateReady:
				ctx2, cancel2 := context.WithTimeout(context.Background(), 3*time.Second)
				info, ierr := c.GetModel(ctx2)
				cancel2()
				if ierr == nil {
					return info, nil
				}
				// Ready but GetModel raced the publish; retry on the next tick.
			case StateFailed:
				return nil, fmt.Errorf("engine failed to start: %s", st.Message)
			}
		}
		if aliveCheck != nil {
			if exitErr := aliveCheck(); exitErr != nil {
				return nil, exitErr
			}
		}
		if time.Since(lastChange) > stallTimeout {
			if haveStatus {
				return nil, fmt.Errorf("engine load stalled at %q (%d/%d) for %s",
					last.Phase, last.Done, last.Total, stallTimeout)
			}
			return nil, fmt.Errorf("engine did not start within %s", stallTimeout)
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
