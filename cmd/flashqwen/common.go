package main

import (
	"time"

	"flashqwen/internal/chatml"
	"flashqwen/internal/engine"
	"flashqwen/internal/supervisor"
	"flashqwen/internal/tokenizer"
)

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
	// Block until the engine serves. sup.Exited makes a dying engine surface its real cause
	// immediately instead of polling until the timeout.
	info, err := conn.Ready(120*time.Second, sup.Exited)
	if err != nil {
		conn.Close()
		sup.Stop()
		return nil, err
	}
	gen := engine.NewGenerator(conn, cm, tok, info.MaxCtx)
	return &session{gen: gen, tok: tok, info: info, stop: func() { conn.Close(); sup.Stop() }}, nil
}
