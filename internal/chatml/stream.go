package chatml

import (
	"encoding/json"
	"strconv"
)

// Stream consumes the engine's token ids for one response and produces visible text deltas and
// completed tool calls. Control tokens are hidden from text; ids between <tool_call> and
// </tool_call> are buffered and parsed into a ToolCall.
type Stream struct {
	m       *Format
	ids     []int // visible (non-special) text ids so far
	prev    string
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
		s.ids = append(s.ids, id)
		full := s.m.tok.Decode(s.ids)
		delta := full[len(s.prev):]
		s.prev = full
		return delta, nil
	}
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
