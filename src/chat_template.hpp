// Qwen3 ChatML rendering. Owns the model-specific prompt format so the gateway stays
// model-agnostic: it sends structured messages + tools, the engine renders the prompt.
#pragma once
#include <string>
#include <vector>

struct ChatToolCall { std::string id, name, arguments_json; };

struct ChatMessage {
    std::string role;                       // system | user | assistant | tool
    std::string content;
    std::vector<ChatToolCall> tool_calls;   // assistant turns that called tools
    std::string tool_call_id;               // tool-result turns
};

struct ChatToolDef { std::string name, description, parameters_json; };

// Render the conversation into Qwen3 ChatML text: an optional system block (carrying the tools
// section when tools are present), the turns (assistant tool_calls as <tool_call> blocks, tool
// results merged into a user <tool_response> turn), then the assistant generation prompt — with
// an empty <think></think> block when thinking is disabled.
std::string render_chatml(const std::vector<ChatMessage>& messages,
                          const std::vector<ChatToolDef>& tools, bool enable_thinking);
