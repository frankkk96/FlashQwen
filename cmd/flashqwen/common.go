package main

import (
	"context"
	"fmt"
	"time"

	"flashqwen/internal/chatml"
	"flashqwen/internal/engine"
	"flashqwen/internal/supervisor"
	"flashqwen/internal/tokenizer"
)

// session bundles the embedded engine client plus its lifecycle.
type session struct {
	eng  *engine.Client
	info *engine.ModelInfo
	stop func()
}

// open loads the tokenizer + chat template from the model directory, launches the embedded engine
// against it, and waits for it to serve. The returned stop() tears both down. maxQueue caps the
// engine's admission queue (<=0 => engine default of 4*slots).
func open(modelDir string, slots, maxCtx, maxQueue int) (*session, error) {
	tok, err := tokenizer.Load(modelDir)
	if err != nil {
		return nil, err
	}
	cm := chatml.Load(modelDir, tok)

	sup, err := supervisor.Start(modelDir, slots, maxCtx, maxQueue)
	if err != nil {
		return nil, err
	}
	eng, err := engine.Dial(sup.Addr, cm, tok)
	if err != nil {
		sup.Stop()
		return nil, err
	}
	info, err := waitReady(eng, sup, 120*time.Second)
	if err != nil {
		eng.Close()
		sup.Stop()
		return nil, err
	}
	eng.SetMaxCtx(info.MaxCtx)
	return &session{eng: eng, info: info, stop: func() { eng.Close(); sup.Stop() }}, nil
}

// waitReady polls GetModel until the engine answers, the engine process dies, or the timeout
// elapses. If the engine exits during startup (bad port bind, model load failure, CUDA error, ...)
// it returns that detailed cause immediately rather than polling a dead process until the timeout.
func waitReady(eng *engine.Client, sup *supervisor.Engine, timeout time.Duration) (*engine.ModelInfo, error) {
	deadline := time.Now().Add(timeout)
	for {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		info, err := eng.GetModel(ctx)
		cancel()
		if err == nil {
			return info, nil
		}
		if exitErr := sup.Exited(); exitErr != nil {
			return nil, exitErr
		}
		if time.Now().After(deadline) {
			return nil, fmt.Errorf("engine did not become ready within %s: %w", timeout, err)
		}
		time.Sleep(500 * time.Millisecond)
	}
}
