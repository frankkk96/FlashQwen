package chatml

import (
	"encoding/json"
	"strconv"
	"unicode/utf8"
)

// Stream consumes the engine's token ids for one response and produces visible text deltas and
// completed tool calls. Control tokens are hidden from text; ids between <tool_call> and
// </tool_call> are buffered and parsed into a ToolCall.
type Stream struct {
	m       *Format
	pending []byte // decoded bytes not yet emitted (the tail may be a half-finished rune)
	inTool  bool
	toolIDs []int
	nTools  int
}

func (m *Format) NewStream() *Stream { return &Stream{m: m} }

// Push feeds one token id. It returns any newly-decoded visible text and/or a completed tool call
// (either may be empty/nil).
func (s *Stream) Push(id int) (text string, tool *ToolCall) {
	switch {
	case id == s.m.toolOpen:
		s.inTool = true
		s.toolIDs = s.toolIDs[:0]
		return "", nil
	case id == s.m.toolClose:
		s.inTool = false
		return "", s.parseTool()
	case s.inTool:
		s.toolIDs = append(s.toolIDs, id)
		return "", nil
	case s.m.tok.IsSpecial(id):
		return "", nil // hide control tokens (im_end, think markers, ...)
	default:
		// Decode is purely concatenative, so per-token decode + accumulate is byte-identical to
		// decoding the whole id list each step (and O(n) instead of O(n²)). A multibyte rune can
		// straddle two tokens, so only emit up to the last complete rune; hold the rest back.
		s.pending = append(s.pending, s.m.tok.Decode([]int{id})...)
		return s.takeRunes(), nil
	}
}

// takeRunes returns the prefix of pending that forms complete UTF-8 runes, retaining any
// incomplete trailing multibyte sequence for the next token.
func (s *Stream) takeRunes() string {
	i := 0
	for i < len(s.pending) {
		if s.pending[i] < utf8.RuneSelf {
			i++ // ASCII
			continue
		}
		if !utf8.FullRune(s.pending[i:]) {
			break // tail is a half-finished multibyte char; wait for more bytes
		}
		_, size := utf8.DecodeRune(s.pending[i:])
		i += size // FullRune true => size>=1, so even invalid encodings advance (no infinite loop)
	}
	if i == 0 {
		return ""
	}
	out := string(s.pending[:i])
	n := copy(s.pending, s.pending[i:]) // shift the 0–3 leftover bytes to the front, reuse the array
	s.pending = s.pending[:n]
	return out
}

// Flush returns any bytes still buffered (a trailing rune truncated at end of stream, e.g. when
// max_tokens lands mid-character). Call it once after the last Push so the aggregated text isn't
// missing those bytes; emitting them as-is matches what a whole-list Decode would have produced.
func (s *Stream) Flush() string {
	if len(s.pending) == 0 {
		return ""
	}
	out := string(s.pending)
	s.pending = s.pending[:0]
	return out
}

func (s *Stream) parseTool() *ToolCall {
	raw := s.m.tok.Decode(s.toolIDs)
	var parsed struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	}
	if json.Unmarshal([]byte(raw), &parsed) != nil || parsed.Name == "" {
		return nil
	}
	args := "{}"
	if len(parsed.Arguments) > 0 {
		args = string(parsed.Arguments)
	}
	tc := &ToolCall{ID: "call_" + strconv.Itoa(s.nTools), Name: parsed.Name, ArgumentsJSON: args}
	s.nTools++
	return tc
}
