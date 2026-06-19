package server

import "encoding/json"

// ---- OpenAI request types ----------------------------------------------------------------

type ChatRequest struct {
	Model         string         `json:"model"`
	Messages      []ChatMessage  `json:"messages"`
	Tools         []Tool         `json:"tools,omitempty"`
	Stream        bool           `json:"stream"`
	StreamOptions *StreamOptions `json:"stream_options"`
	MaxTokens     int            `json:"max_tokens"`
	Temperature   *float64       `json:"temperature"`
	TopP          *float64       `json:"top_p"`
	// vLLM extension: generate exactly max_tokens, ignoring EOS / stop tokens (benchmarking).
	IgnoreEOS bool `json:"ignore_eos"`
	// FlashQwen extension: Qwen3 thinking mode (default off).
	EnableThinking *bool `json:"enable_thinking"`
}

// CompletionRequest is the OpenAI /v1/completions wire type — raw text in, raw text out, no chat
// template. prompt is a string or an array of token ids (arrays of prompts are not supported).
// ignore_eos is a vLLM extension used by benchmarks to force exactly max_tokens of output.
type CompletionRequest struct {
	Model         string          `json:"model"`
	Prompt        json.RawMessage `json:"prompt"`
	MaxTokens     int             `json:"max_tokens"`
	Temperature   *float64        `json:"temperature"`
	TopP          *float64        `json:"top_p"`
	Stream        bool            `json:"stream"`
	StreamOptions *StreamOptions  `json:"stream_options"`
	IgnoreEOS     bool            `json:"ignore_eos"`
}

type CompletionChoice struct {
	Index        int     `json:"index"`
	Text         string  `json:"text"`
	FinishReason *string `json:"finish_reason"`
}

type Completion struct {
	ID      string             `json:"id"`
	Object  string             `json:"object"` // "text_completion"
	Created int64              `json:"created"`
	Model   string             `json:"model"`
	Choices []CompletionChoice `json:"choices"`
	Usage   *Usage             `json:"usage,omitempty"`
}

// StreamOptions mirrors OpenAI's stream_options. When IncludeUsage is set, the stream ends with an
// extra chunk carrying token usage and an empty choices array, before the final [DONE].
type StreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type ChatMessage struct {
	Role       string          `json:"role"`
	Content    json.RawMessage `json:"content"` // string, or [{type,text}], or null
	Name       string          `json:"name,omitempty"`
	ToolCalls  []ToolCall      `json:"tool_calls,omitempty"`
	ToolCallID string          `json:"tool_call_id,omitempty"`
}

// Text flattens an OpenAI content field (a plain string, a multimodal parts array, or null) into
// plain text — the only thing a text model consumes.
func (m ChatMessage) Text() string {
	if len(m.Content) == 0 || string(m.Content) == "null" {
		return ""
	}
	var s string
	if err := json.Unmarshal(m.Content, &s); err == nil {
		return s
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(m.Content, &parts); err == nil {
		out := ""
		for _, p := range parts {
			if p.Type == "text" {
				out += p.Text
			}
		}
		return out
	}
	return ""
}

type Tool struct {
	Type     string      `json:"type"`
	Function FunctionDef `json:"function"`
}

type FunctionDef struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

type ToolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"` // "function"
	Function ToolCallFunction `json:"function"`
}

type ToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"` // a JSON-encoded string, per the OpenAI spec
}

// ---- OpenAI response types ---------------------------------------------------------------

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type RespMessage struct {
	Role      string     `json:"role"`
	Content   *string    `json:"content"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

type Choice struct {
	Index        int          `json:"index"`
	Message      *RespMessage `json:"message,omitempty"` // non-streaming
	Delta        *RespMessage `json:"delta,omitempty"`   // streaming
	FinishReason *string      `json:"finish_reason"`
}

type ChatCompletion struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   *Usage   `json:"usage,omitempty"`
}

// ---- /v1/models --------------------------------------------------------------------------

type Model struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

type ModelList struct {
	Object string  `json:"object"`
	Data   []Model `json:"data"`
}
