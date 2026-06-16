package llm

import "flashqwen/internal/chat"

// Request is a format-neutral completion request. Format adapters (openai, anthropic, ...) decode
// their own wire types into this; the engine-facing details (tokenisation, stop ids, pb) live below
// it in Service.
type Request struct {
	Messages       []chat.Message
	Tools          []chat.ToolDef
	EnableThinking bool
	MaxTokens      int      // 0 => default
	Temperature    *float64 // nil => engine default (greedy)
	TopP           *float64 // nil => off (1.0)
}

// Result is the fully-aggregated response. Streaming adapters also receive the same text/tool
// pieces incrementally via the onDelta callback to Service.Generate; blocking adapters just read
// this.
type Result struct {
	Text             string
	ToolCalls        []chat.ToolCall
	FinishReason     string // "stop" | "length" | "tool_calls"
	PromptTokens     int
	CompletionTokens int
}
