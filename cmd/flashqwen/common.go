package main

import (
	"context"
	"fmt"
	"time"

	"flashqwen/internal/chat"
	pb "flashqwen/internal/enginepb"
	"flashqwen/internal/engine"
	"flashqwen/internal/supervisor"
	"flashqwen/internal/tokenizer"
)

// session bundles the embedded engine plus the Go-side model-text layer.
type session struct {
	eng  *engine.Client
	cm   *chat.Model
	tok  *tokenizer.Tokenizer
	info *pb.ModelInfo
	stop func()
}

// open loads the tokenizer + chat template, launches the embedded engine against modelDir, and
// waits for it to serve. The returned stop() tears both down.
func open(modelDir string, slots, maxCtx int) (*session, error) {
	tok, err := tokenizer.Load(modelDir)
	if err != nil {
		return nil, err
	}
	cm := chat.Load(modelDir, tok)

	sup, err := supervisor.Start(modelDir, slots, maxCtx)
	if err != nil {
		return nil, err
	}
	cli, err := engine.Dial(sup.Addr)
	if err != nil {
		sup.Stop()
		return nil, err
	}
	info, err := waitReady(cli, 120*time.Second)
	if err != nil {
		cli.Close()
		sup.Stop()
		return nil, fmt.Errorf("engine did not become ready: %w", err)
	}
	return &session{eng: cli, cm: cm, tok: tok, info: info, stop: func() { cli.Close(); sup.Stop() }}, nil
}

// waitReady polls GetModel until the engine answers or the timeout elapses.
func waitReady(cli *engine.Client, timeout time.Duration) (*pb.ModelInfo, error) {
	deadline := time.Now().Add(timeout)
	for {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		info, err := cli.GetModel(ctx)
		cancel()
		if err == nil {
			return info, nil
		}
		if time.Now().After(deadline) {
			return nil, err
		}
		time.Sleep(500 * time.Millisecond)
	}
}

func (s *session) maxCtx() int { return int(s.info.MaxCtx) }
