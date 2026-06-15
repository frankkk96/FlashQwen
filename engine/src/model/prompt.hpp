// Qwen3 ChatML prompt construction — the single source of truth for the model's prompt format.
// Shared by the interactive CLI (src/chat.cpp, incremental per-turn) and the gRPC server
// (src/grpc_server.cpp, full-conversation render), so the format literals live in exactly one place.
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

// ChatML building blocks (the actual markers live only here).
void append_user_turn(std::string& out, const std::string& content);          // <|im_start|>user ... <|im_end|>
void append_assistant_header(std::string& out, bool enable_thinking);         // <|im_start|>assistant + optional empty <think>

// Full Qwen3 prompt from a structured conversation (+ tools): an optional system block (carrying the
// tools section when tools are present), the turns (assistant tool_calls as <tool_call> blocks, tool
// results merged into a user <tool_response> turn), then the assistant generation header.
std::string render_prompt(const std::vector<ChatMessage>& messages,
                          const std::vector<ChatToolDef>& tools, bool enable_thinking);
