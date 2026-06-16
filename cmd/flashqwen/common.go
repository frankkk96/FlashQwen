package main

import (
	"time"

	"flashqwen/internal/chatml"
	"flashqwen/internal/engine"
	"flashqwen/internal/supervisor"
	"flashqwen/internal/tokenizer"
)

// session bundles the embedded engine plus its lifecycle. conn is the low-level token transport
// (used by benchmark); gen is the text tier on top of it (used by serve / chat).
type session struct {
	conn *engine.Client
	gen  *engine.Generator
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
	return &session{conn: conn, gen: gen, info: info, stop: func() { conn.Close(); sup.Stop() }}, nil
}
