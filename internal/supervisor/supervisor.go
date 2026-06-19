// Package supervisor embeds the compiled C++ token engine, extracts it at runtime, spawns it on a
// localhost gRPC port, and manages its lifecycle. The Go binary is thus self-launching: no separate
// engine process to start. (The extracted engine still needs CUDA / gRPC shared libs installed —
// embedding the executable does not embed its .so dependencies.)
package supervisor

import (
	_ "embed"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"syscall"
)

//go:embed bin/flashqwen-engine
var engineBinary []byte

type Engine struct {
	Addr string // host:port the engine's gRPC server listens on
	cmd  *exec.Cmd
	path string // extracted binary path (removed on Stop)

	tail    *tailBuffer   // last few KB of engine stderr (for error reporting)
	done    chan struct{} // closed once the process has been reaped
	waitErr error         // result of cmd.Wait(); set before done is closed
}

// tailBuffer is an io.Writer that retains only the last `max` bytes written — used to keep the
// tail of engine stderr around so a startup failure can be reported with its actual cause.
type tailBuffer struct {
	mu  sync.Mutex
	buf []byte
	max int
}

func (t *tailBuffer) Write(p []byte) (int, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.buf = append(t.buf, p...)
	if len(t.buf) > t.max {
		t.buf = t.buf[len(t.buf)-t.max:]
	}
	return len(p), nil
}

func (t *tailBuffer) String() string {
	t.mu.Lock()
	defer t.mu.Unlock()
	return string(t.buf)
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
	// Forward engine stderr to the terminal so the user sees its load progress + logs, while also
	// retaining the tail so a startup failure can be reported with the engine's actual cause.
	tail := &tailBuffer{max: 8 << 10}
	cmd.Stderr = io.MultiWriter(os.Stderr, tail)
	if err := cmd.Start(); err != nil {
		os.Remove(f.Name())
		return nil, fmt.Errorf("start engine: %w", err)
	}
	e := &Engine{Addr: addr, cmd: cmd, path: f.Name(), tail: tail, done: make(chan struct{})}
	go func() { e.waitErr = cmd.Wait(); close(e.done) }() // reap; Exited()/Stop() observe via done
	return e, nil
}

// Exited reports nil while the engine is still running, or a detailed error once it has terminated:
// the exit status plus the tail of its stderr. waitReady polls this so a dying engine surfaces its
// real cause immediately instead of after the readiness timeout.
func (e *Engine) Exited() error {
	select {
	case <-e.done:
		msg := "engine process exited"
		if e.waitErr != nil {
			msg += " (" + e.waitErr.Error() + ")"
		}
		if t := strings.TrimSpace(e.tail.String()); t != "" {
			msg += "; engine stderr:\n" + t
		}
		return errors.New(msg)
	default:
		return nil
	}
}

// Stop signals the engine to terminate and cleans up the extracted binary.
func (e *Engine) Stop() {
	if e.cmd.Process != nil {
		_ = e.cmd.Process.Signal(syscall.SIGTERM)
		<-e.done // the reaper goroutine owns cmd.Wait(); block until the process is gone
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
