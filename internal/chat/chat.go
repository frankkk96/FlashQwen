// Package chat owns the model-text layer the C++ engine no longer has: the Qwen3 ChatML prompt
// format, the stop (eos) token ids, and token-level tool-call detection over the engine's id
// stream. It sits between the OpenAI HTTP layer and the token engine.
//
// The ChatML template is hand-ported here rather than rendered from tokenizer_config.json's Jinja:
// the Qwen3 chat_template uses constructs (namespace, reverse slicing, method chains) that the Go
// Jinja engines can't parse. This builder matches the common OpenAI surface (system/user/assistant
// /tool turns, tool definitions, tool calls, thinking toggle).
package chat

import (
	"encoding/json"
	"os"
	"strings"

	"flashqwen/internal/tokenizer"
)

// Neutral message/tool types (the OpenAI layer maps its own types into these).
type ToolCall struct {
	ID            string
	Name          string
	ArgumentsJSON string // arguments as a JSON object string
}

type Message struct {
	Role       string // system | user | assistant | tool
	Content    string
	ToolCalls  []ToolCall // assistant turns that called tools
	ToolCallID string     // tool-result turns
}

type ToolDef struct {
	Name           string
	Description    string
	ParametersJSON string // JSON Schema, as a string
}

type Model struct {
	tok       *tokenizer.Tokenizer
	eosIDs    []int
	toolOpen  int // id of "<tool_call>"  (-1 if absent)
	toolClose int // id of "</tool_call>"
}

// Load resolves the model-text constants: eos ids (from generation_config.json, falling back to
// the canonical Qwen3 ids) and the <tool_call> markers.
func Load(dir string, tok *tokenizer.Tokenizer) *Model {
	m := &Model{tok: tok, toolOpen: -1, toolClose: -1}
	if id, ok := tok.ID("<tool_call>"); ok {
		m.toolOpen = id
	}
	if id, ok := tok.ID("</tool_call>"); ok {
		m.toolClose = id
	}
	m.eosIDs = eosFromGenerationConfig(dir)
	if len(m.eosIDs) == 0 { // fallback: canonical Qwen3 eos (<|im_end|>, <|endoftext|>)
		for _, s := range []string{"<|im_end|>", "<|endoftext|>"} {
			if id, ok := tok.ID(s); ok {
				m.eosIDs = append(m.eosIDs, id)
			}
		}
	}
	return m
}

// eosFromGenerationConfig reads eos_token_id from <dir>/generation_config.json — it may be a single
// int or a list of ints. Returns nil if the file is absent/unparseable (Load then falls back).
func eosFromGenerationConfig(dir string) []int {
	b, err := os.ReadFile(dir + "/generation_config.json")
	if err != nil {
		return nil
	}
	var cfg struct {
		EOS json.RawMessage `json:"eos_token_id"`
	}
	if json.Unmarshal(b, &cfg) != nil || len(cfg.EOS) == 0 {
		return nil
	}
	var list []int
	if json.Unmarshal(cfg.EOS, &list) == nil {
		return list
	}
	var one int
	if json.Unmarshal(cfg.EOS, &one) == nil {
		return []int{one}
	}
	return nil
}

// StopTokenIDs are the ids the engine should stop on.
func (m *Model) StopTokenIDs() []int32 {
	out := make([]int32, len(m.eosIDs))
	for i, id := range m.eosIDs {
		out[i] = int32(id)
	}
	return out
}

// ---- ChatML prompt construction ---------------------------------------------------------

func toolJSON(t ToolDef) string {
	params := json.RawMessage(t.ParametersJSON)
	if len(params) == 0 {
		params = json.RawMessage("{}")
	}
	obj := map[string]any{
		"type": "function",
		"function": map[string]any{
			"name":       t.Name,
			"parameters": params,
		},
	}
	if t.Description != "" {
		obj["function"].(map[string]any)["description"] = t.Description
	}
	b, _ := json.Marshal(obj)
	return string(b)
}

// Render builds the full Qwen3 ChatML prompt for a conversation (+ tools), ending with the
// assistant generation header (plus an empty <think></think> block when thinking is off).
func (m *Model) Render(msgs []Message, tools []ToolDef, enableThinking bool) string {
	var b strings.Builder
	hasSystem := len(msgs) > 0 && msgs[0].Role == "system"

	if len(tools) > 0 {
		b.WriteString("<|im_start|>system\n")
		if hasSystem {
			b.WriteString(msgs[0].Content)
			b.WriteString("\n\n")
		}
		b.WriteString("# Tools\n\nYou may call one or more functions to assist with the user query.\n\n")
		b.WriteString("You are provided with function signatures within <tools></tools> XML tags:\n<tools>")
		for _, t := range tools {
			b.WriteString("\n")
			b.WriteString(toolJSON(t))
		}
		b.WriteString("\n</tools>\n\nFor each function call, return a json object with function name and " +
			"arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n" +
			"{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n")
	} else if hasSystem {
		b.WriteString("<|im_start|>system\n")
		b.WriteString(msgs[0].Content)
		b.WriteString("<|im_end|>\n")
	}

	start := 0
	if hasSystem {
		start = 1
	}
	for i := start; i < len(msgs); i++ {
		mm := msgs[i]
		switch mm.Role {
		case "user":
			b.WriteString("<|im_start|>user\n")
			b.WriteString(mm.Content)
			b.WriteString("<|im_end|>\n")
		case "assistant":
			b.WriteString("<|im_start|>assistant\n")
			b.WriteString(mm.Content)
			for _, tc := range mm.ToolCalls {
				args := tc.ArgumentsJSON
				if args == "" {
					args = "{}"
				}
				b.WriteString("\n<tool_call>\n{\"name\": \"")
				b.WriteString(tc.Name)
				b.WriteString("\", \"arguments\": ")
				b.WriteString(args)
				b.WriteString("}\n</tool_call>")
			}
			b.WriteString("<|im_end|>\n")
		case "tool":
			// Merge consecutive tool results into one user turn of <tool_response> blocks.
			if i == start || msgs[i-1].Role != "tool" {
				b.WriteString("<|im_start|>user\n")
			}
			b.WriteString("<tool_response>\n")
			b.WriteString(mm.Content)
			b.WriteString("\n</tool_response>\n")
			if i+1 >= len(msgs) || msgs[i+1].Role != "tool" {
				b.WriteString("<|im_end|>\n")
			}
		}
	}

	b.WriteString("<|im_start|>assistant\n")
	if !enableThinking {
		b.WriteString("<think>\n\n</think>\n\n")
	}
	return b.String()
}

// ---- token-stream decoding + tool-call detection ----------------------------------------

// Stream consumes the engine's token ids for one response and produces visible text deltas and
// completed tool calls. Control tokens are hidden from text; ids between <tool_call> and
// </tool_call> are buffered and parsed into a ToolCall.
type Stream struct {
	m       *Model
	ids     []int // visible (non-special) text ids so far
	prev    string
	inTool  bool
	toolIDs []int
	nTools  int
}

func (m *Model) NewStream() *Stream { return &Stream{m: m} }

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
	tc := &ToolCall{ID: "call_" + itoa(s.nTools), Name: parsed.Name, ArgumentsJSON: args}
	s.nTools++
	return tc
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var d []byte
	for n > 0 {
		d = append([]byte{byte('0' + n%10)}, d...)
		n /= 10
	}
	return string(d)
}
