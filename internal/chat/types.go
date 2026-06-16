package chat

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
