package main

import (
	"encoding/json"
	"fmt"
	"strings"
)

// extractToolCalls pulls Qwen3 <tool_call>{"name":..,"arguments":..}</tool_call> blocks out of a
// completion and converts them to OpenAI tool_calls. It returns the parsed calls and the text with
// those blocks removed (the remaining natural-language content, trimmed).
func extractToolCalls(text string) ([]ToolCall, string) {
	const open, close = "<tool_call>", "</tool_call>"
	var calls []ToolCall
	var clean strings.Builder
	rest := text
	for {
		i := strings.Index(rest, open)
		if i < 0 {
			clean.WriteString(rest)
			break
		}
		clean.WriteString(rest[:i])
		rest = rest[i+len(open):]
		j := strings.Index(rest, close)
		var body string
		if j < 0 {
			body = rest // unterminated (e.g. truncated by length) — best-effort parse
			rest = ""
		} else {
			body = rest[:j]
			rest = rest[j+len(close):]
		}
		var parsed struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		}
		if err := json.Unmarshal([]byte(strings.TrimSpace(body)), &parsed); err != nil || parsed.Name == "" {
			continue // not a well-formed call; drop it
		}
		args := string(parsed.Arguments)
		if args == "" {
			args = "{}"
		}
		calls = append(calls, ToolCall{
			ID:       fmt.Sprintf("call_%d", len(calls)),
			Type:     "function",
			Function: ToolCallFunction{Name: parsed.Name, Arguments: args},
		})
	}
	return calls, strings.TrimSpace(clean.String())
}
