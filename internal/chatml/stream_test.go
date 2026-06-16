package chatml

import (
	"strings"
	"testing"
	"unicode/utf8"
)

// feed mirrors stream.go's default branch (append a token's raw bytes, then take complete runes),
// minus the tokenizer lookup — letting us drive the rune-boundary logic with explicit byte chunks.
// It returns the deltas a streaming caller would see, including the final Flush.
func feed(s *Stream, chunks [][]byte) []string {
	var deltas []string
	for _, c := range chunks {
		s.pending = append(s.pending, c...)
		if d := s.takeRunes(); d != "" {
			deltas = append(deltas, d)
		}
	}
	if tail := s.Flush(); tail != "" {
		deltas = append(deltas, tail)
	}
	return deltas
}

// A rune that straddles two tokens must never be emitted as half a UTF-8 sequence, and the deltas
// must still reassemble to the exact byte stream (no loss, no corruption).
func TestStreamRuneBoundaries(t *testing.T) {
	cases := []struct {
		name   string
		chunks [][]byte
	}{
		{"ascii", [][]byte{[]byte("he"), []byte("llo")}},
		{"cjk split across two tokens", [][]byte{{0xe4, 0xb8}, {0xad}}},                           // 中
		{"cjk byte by byte", [][]byte{{0xe4}, {0xb8}, {0xad}}},                                    // 中
		{"emoji split", [][]byte{{0xf0, 0x9f}, {0x98, 0x80}}},                                     // 😀 (4 bytes)
		{"ascii + cjk straddle", [][]byte{[]byte("a"), {0xe4, 0xb8}, {0xad, 0xe5}, {0xa5, 0xbd}}}, // a中好
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var want []byte
			for _, c := range tc.chunks {
				want = append(want, c...)
			}
			deltas := feed(&Stream{}, tc.chunks)
			for i, d := range deltas { // (a) no mid-stream split-rune corruption
				if !utf8.ValidString(d) {
					t.Errorf("delta %d %q is not valid UTF-8", i, d)
				}
			}
			if got := strings.Join(deltas, ""); got != string(want) { // (b) lossless round-trip
				t.Errorf("reassembled %q, want %q", got, string(want))
			}
		})
	}
}

// When the stream ends mid-character (e.g. max_tokens lands inside a multibyte rune), takeRunes
// holds the partial bytes back; Flush must still surface them so the aggregated text isn't lossy.
func TestStreamFlushTruncated(t *testing.T) {
	s := &Stream{}
	s.pending = append(s.pending, 0xe4, 0xb8) // first 2 of the 3 bytes of 中, then the stream ends
	if d := s.takeRunes(); d != "" {
		t.Fatalf("takeRunes emitted %q before the rune was complete", d)
	}
	if tail := s.Flush(); tail != string([]byte{0xe4, 0xb8}) {
		t.Errorf("Flush = %q, want the 2 held-back bytes", tail)
	}
	if s.Flush() != "" {
		t.Error("second Flush should be empty")
	}
}
