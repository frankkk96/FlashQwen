// Package chatml is the Qwen3 ChatML format layer the C++ engine no longer has: a bidirectional
// codec sitting between the neutral message types and the token engine. It renders messages +
// tools into a ChatML prompt (inbound) and decodes the engine's id stream back into text + tool
// calls (outbound), and it resolves the model's stop (eos) token ids.
//
// The package is split by role: types.go (neutral message/tool vocabulary), render.go (inbound:
// messages -> ChatML prompt), stream.go (outbound: token stream -> text + tool calls), and this
// file (the Format entry point and its load-time constants).
//
// The ChatML template is hand-ported in render.go rather than rendered from tokenizer_config.json's
// Jinja: the Qwen3 chat_template uses constructs (namespace, reverse slicing, method chains) that
// the Go Jinja engines can't parse. It matches the common OpenAI surface (system/user/assistant/
// tool turns, tool definitions, tool calls, thinking toggle).
package chatml

import (
	"encoding/json"
	"os"

	"flashqwen/internal/tokenizer"
)

type Format struct {
	tok       *tokenizer.Tokenizer
	eosIDs    []int
	toolOpen  int // id of "<tool_call>"  (-1 if absent)
	toolClose int // id of "</tool_call>"
}

// Load resolves the model-text constants: eos ids (from generation_config.json, falling back to
// the canonical Qwen3 ids) and the <tool_call> markers.
func Load(dir string, tok *tokenizer.Tokenizer) *Format {
	m := &Format{tok: tok, toolOpen: -1, toolClose: -1}
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
func (m *Format) StopTokenIDs() []int32 {
	out := make([]int32, len(m.eosIDs))
	for i, id := range m.eosIDs {
		out[i] = int32(id)
	}
	return out
}
