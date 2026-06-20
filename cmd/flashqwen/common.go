package main

import (
	"time"

	"flashqwen/internal/chatml"
	"flashqwen/internal/engine"
	"flashqwen/internal/supervisor"
	"flashqwen/internal/tokenizer"
)

// startupTimeout caps how long we wait for the engine to load the model and start serving before
// giving up. The engine loads before binding its port, so this is the whole load budget; generous so
// a large model on a cold cache is never killed prematurely.
const startupTimeout = 1800 * time.Second

// session bundles the embedded engine plus its lifecycle. gen is the text tier (used by serve /
// chat); info carries the engine's reported model metadata; stop tears the engine down.
type session struct {
	gen  *engine.Generator
	info *engine.ModelInfo
	stop func()
}

// open loads the tokenizer + chat template from the model directory, launches the embedded engine
// against it, and waits for it to serve. The returned stop() tears both down. maxQueue caps the
// engine's admission queue (<=0 => engine default of 4*slots).
func open(modelDir string, slots, maxCtx, maxQueue, maxBatchTokens, maxPrefill int, gpuMemFraction float64) (*session, error) {
	tok, err := tokenizer.Load(modelDir)
	if err != nil {
		return nil, err
	}
	cm := chatml.Load(modelDir, tok)

	sup, err := supervisor.Start(modelDir, slots, maxCtx, maxQueue, maxBatchTokens, maxPrefill, gpuMemFraction)
	if err != nil {
		return nil, err
	}
	conn, err := engine.Dial(sup.Addr)
	if err != nil {
		sup.Stop()
		return nil, err
	}
	// Block until the engine has loaded the model and started serving (its load progress is logged to
	// the terminal via the engine's stderr). sup.Exited makes a dying engine surface its real cause
	// immediately instead of waiting out the timeout.
	info, err := conn.WaitReady(startupTimeout, sup.Exited)
	if err != nil {
		conn.Close()
		sup.Stop()
		return nil, err
	}
	gen := engine.NewGenerator(conn, cm, tok, info.MaxCtx)
	return &session{gen: gen, info: info, stop: func() { conn.Close(); sup.Stop() }}, nil
}
