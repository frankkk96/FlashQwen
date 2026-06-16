// Package supervisor embeds the compiled C++ token engine, extracts it at runtime, spawns it on a
// localhost gRPC port, and manages its lifecycle. The Go binary is thus self-launching: no separate
// engine process to start. (The extracted engine still needs CUDA / gRPC shared libs installed —
// embedding the executable does not embed its .so dependencies.)
package supervisor

import (
	_ "embed"
	"fmt"
	"net"
	"os"
	"os/exec"
	"strconv"
	"syscall"
)

//go:embed bin/flashqwen-engine
var engineBinary []byte

type Engine struct {
	Addr string // host:port the engine's gRPC server listens on
	cmd  *exec.Cmd
	path string // extracted binary path (removed on Stop)
}

// Start extracts and launches the engine against modelDir, returning once the process has been
// spawned (call WaitReady on a client to block until it serves). Engine stderr is forwarded.
// maxQueue caps how many requests may wait for admission before the engine rejects new ones as
// over-capacity; <=0 lets the engine pick its default (4*slots).
func Start(modelDir string, slots, maxCtx, maxQueue int) (*Engine, error) {
	f, err := os.CreateTemp("", "flashqwen-engine-*")
	if err != nil {
		return nil, err
	}
	if _, err := f.Write(engineBinary); err != nil {
		f.Close()
		return nil, err
	}
	f.Close()
	if err := os.Chmod(f.Name(), 0o755); err != nil {
		return nil, err
	}

	addr, err := freePort()
	if err != nil {
		os.Remove(f.Name())
		return nil, err
	}

	cmd := exec.Command(f.Name(),
		"--model", modelDir,
		"--address", addr,
		"--slots", strconv.Itoa(slots),
		"--max-ctx", strconv.Itoa(maxCtx),
		"--max-queue", strconv.Itoa(maxQueue))
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		os.Remove(f.Name())
		return nil, fmt.Errorf("start engine: %w", err)
	}
	return &Engine{Addr: addr, cmd: cmd, path: f.Name()}, nil
}

// Stop signals the engine to terminate and cleans up the extracted binary.
func (e *Engine) Stop() {
	if e.cmd.Process != nil {
		_ = e.cmd.Process.Signal(syscall.SIGTERM)
		_ = e.cmd.Wait()
	}
	os.Remove(e.path)
}

// freePort asks the OS for an unused localhost port (closed immediately, then handed to the engine).
func freePort() (string, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return "", err
	}
	defer l.Close()
	return l.Addr().String(), nil
}
