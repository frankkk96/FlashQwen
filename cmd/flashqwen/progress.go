package main

import (
	"fmt"
	"os"
	"strings"

	"flashqwen/internal/engine"
)

// startupBar renders engine load progress as a single in-place (\r) updating line on stderr. The
// engine's own stderr is buffered (not forwarded to the terminal) during startup, so this owns the
// line without interleaving.
type startupBar struct{ active bool }

func (b *startupBar) update(s engine.Status) {
	if s.State != engine.StateLoading {
		return
	}
	b.active = true
	if s.Total > 0 {
		const width = 28
		filled := s.Done * width / s.Total
		if filled > width {
			filled = width
		}
		fmt.Fprintf(os.Stderr, "\r  %-18s [%s%s] %d/%d   ",
			s.Phase, strings.Repeat("=", filled), strings.Repeat(" ", width-filled), s.Done, s.Total)
	} else {
		fmt.Fprintf(os.Stderr, "\r  %-18s ...   ", s.Phase)
	}
}

// finish clears the progress line once loading is over (ready or failed).
func (b *startupBar) finish() {
	if b.active {
		fmt.Fprintf(os.Stderr, "\r%s\r", strings.Repeat(" ", 64))
		b.active = false
	}
}
