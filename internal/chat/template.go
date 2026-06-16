package chat

import (
	"encoding/json"
	"strings"
)

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
