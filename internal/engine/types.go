package engine

import "flashqwen/internal/chatml"

// Request is a format-neutral completion request. The HTTP/format layer (internal/server) decodes
// its wire types into this; everything below — tokenisation, stop ids, pb — is handled inside the
// Client.
type Request struct {
	Messages       []chatml.Message
	Tools          []chatml.ToolDef
	EnableThinking bool
	MaxTokens      int      // 0 => fill the remaining context window
	Temperature    *float64 // nil => engine default (greedy)
	TopP           *float64 // nil => off (1.0)
}

// Result is the fully-aggregated text response. Streaming callers also receive the same pieces
// incrementally via the onDelta callback to Generate; blocking callers just read this.
type Result struct {
	Text             string
	ToolCalls        []chatml.ToolCall
	FinishReason     string // "stop" | "length" | "tool_calls"
	PromptTokens     int
	CompletionTokens int
}

// Stats is the terminal info from a low-level token Stream (no decoded text).
type Stats struct {
	FinishReason     string
	PromptTokens     int
	CompletionTokens int
}

// ModelInfo is the engine's self-reported metadata (authoritative after KV/context rounding).
type ModelInfo struct {
	ID        string
	MaxCtx    int
	VocabSize int
}
