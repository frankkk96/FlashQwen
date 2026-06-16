package main

import (
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
	// Block until the engine serves (arming the client's context window). sup.Exited makes a dying
	// engine surface its real cause immediately instead of polling until the timeout.
	info, err := eng.Ready(120*time.Second, sup.Exited)
	if err != nil {
		eng.Close()
		sup.Stop()
		return nil, err
	}
	return &session{eng: eng, info: info, stop: func() { eng.Close(); sup.Stop() }}, nil
}
