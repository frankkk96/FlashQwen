package main

import (
	"encoding/json"
	"strings"
)

// renderPrompt turns an OpenAI message list (+ tools) into the Qwen3 ChatML text the engine
// tokenises. It mirrors the Qwen3 HF chat template: an optional system block (carrying the tools
// section when tools are present), the conversation turns (assistant tool_calls rendered as
// <tool_call> blocks, tool results merged into a user <tool_response> turn), and finally the
// assistant generation prompt — with an empty <think></think> block when thinking is disabled.
func renderPrompt(req ChatRequest) string {
	var b strings.Builder

	var systemText string
	hasSystem := false
	if len(req.Messages) > 0 && req.Messages[0].Role == "system" {
		systemText = req.Messages[0].Text()
		hasSystem = true
	}

	if len(req.Tools) > 0 {
		b.WriteString("<|im_start|>system\n")
		if hasSystem {
			b.WriteString(systemText)
			b.WriteString("\n\n")
		}
		b.WriteString("# Tools\n\nYou may call one or more functions to assist with the user query.\n\n")
		b.WriteString("You are provided with function signatures within <tools></tools> XML tags:\n<tools>")
		for _, t := range req.Tools {
			j, _ := json.Marshal(t)
			b.WriteString("\n")
			b.Write(j)
		}
		b.WriteString("\n</tools>\n\nFor each function call, return a json object with function name " +
			"and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n" +
			"{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n")
	} else if hasSystem {
		b.WriteString("<|im_start|>system\n")
		b.WriteString(systemText)
		b.WriteString("<|im_end|>\n")
	}

	msgs := req.Messages
	start := 0
	if hasSystem {
		start = 1
	}
	for i := start; i < len(msgs); i++ {
		m := msgs[i]
		switch m.Role {
		case "user":
			b.WriteString("<|im_start|>user\n")
			b.WriteString(m.Text())
			b.WriteString("<|im_end|>\n")
		case "assistant":
			b.WriteString("<|im_start|>assistant\n")
			b.WriteString(m.Text())
			for _, tc := range m.ToolCalls {
				args := tc.Function.Arguments
				if args == "" {
					args = "{}"
				}
				b.WriteString("\n<tool_call>\n{\"name\": \"")
				b.WriteString(tc.Function.Name)
				b.WriteString("\", \"arguments\": ")
				b.WriteString(args)
				b.WriteString("}\n</tool_call>")
			}
			b.WriteString("<|im_end|>\n")
		case "tool":
			// Merge consecutive tool results into a single user turn of <tool_response> blocks.
			if i == start || msgs[i-1].Role != "tool" {
				b.WriteString("<|im_start|>user\n")
			}
			b.WriteString("<tool_response>\n")
			b.WriteString(m.Text())
			b.WriteString("\n</tool_response>\n")
			if i == len(msgs)-1 || msgs[i+1].Role != "tool" {
				b.WriteString("<|im_end|>\n")
			}
		}
	}

	b.WriteString("<|im_start|>assistant\n")
	if req.EnableThinking == nil || !*req.EnableThinking {
		b.WriteString("<think>\n\n</think>\n\n")
	}
	return b.String()
}
