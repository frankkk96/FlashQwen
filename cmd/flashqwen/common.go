package main

import (
	"time"

	"flashqwen/internal/chatml"
	"flashqwen/internal/engine"
	"flashqwen/internal/supervisor"
	"flashqwen/internal/tokenizer"
)

// startupStallTimeout is how long WaitReady tolerates no load progress (process still alive) before
// giving up. It is a watchdog against a stuck load, not a cap on total load time — a model that
// keeps making progress, however slowly, is never killed.
const startupStallTimeout = 120 * time.Second

// session bundles the embedded engine plus its lifecycle. gen is the text tier (used by serve /
// chat); tok is the same tokenizer, exposed so the benchmark can synthesise prompts of a target
// length without loading it a second time.
type session struct {
	gen  *engine.Generator
	tok  *tokenizer.Tokenizer
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
	conn, err := engine.Dial(sup.Addr)
	if err != nil {
		sup.Stop()
		return nil, err
	}
	// Block until the engine serves, showing a load progress bar. sup.Exited makes a dying engine
	// surface its real cause immediately; WaitReady's stall watchdog tolerates a slow load of a large
	// model (it only gives up if progress stops advancing for startupStallTimeout).
	var bar startupBar
	info, err := conn.WaitReady(startupStallTimeout, sup.Exited, bar.update)
	bar.finish()
	if err != nil {
		conn.Close()
		sup.Stop()
		return nil, err
	}
	gen := engine.NewGenerator(conn, cm, tok, info.MaxCtx)
	return &session{gen: gen, tok: tok, info: info, stop: func() { conn.Close(); sup.Stop() }}, nil
}
