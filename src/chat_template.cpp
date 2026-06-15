#include "chat_template.hpp"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/writer.h"

// Serialize one tool as the OpenAI tool object Qwen3 expects inside <tools>:
//   {"type":"function","function":{"name":..,"description":..,"parameters":<schema>}}
static std::string tool_json(const ChatToolDef& t) {
    rapidjson::StringBuffer sb;
    rapidjson::Writer<rapidjson::StringBuffer> w(sb);
    w.StartObject();
    w.Key("type"); w.String("function");
    w.Key("function"); w.StartObject();
    w.Key("name"); w.String(t.name.c_str(), (rapidjson::SizeType)t.name.size());
    if (!t.description.empty()) {
        w.Key("description"); w.String(t.description.c_str(), (rapidjson::SizeType)t.description.size());
    }
    w.Key("parameters");
    if (t.parameters_json.empty()) w.StartObject(), w.EndObject();
    else w.RawValue(t.parameters_json.c_str(), t.parameters_json.size(), rapidjson::kObjectType);
    w.EndObject();
    w.EndObject();
    return std::string(sb.GetString(), sb.GetSize());
}

std::string render_chatml(const std::vector<ChatMessage>& msgs,
                          const std::vector<ChatToolDef>& tools, bool enable_thinking) {
    std::string b;
    bool has_system = !msgs.empty() && msgs[0].role == "system";

    if (!tools.empty()) {
        b += "<|im_start|>system\n";
        if (has_system) { b += msgs[0].content; b += "\n\n"; }
        b += "# Tools\n\nYou may call one or more functions to assist with the user query.\n\n";
        b += "You are provided with function signatures within <tools></tools> XML tags:\n<tools>";
        for (const auto& t : tools) { b += "\n"; b += tool_json(t); }
        b += "\n</tools>\n\nFor each function call, return a json object with function name and "
             "arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n"
             "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n";
    } else if (has_system) {
        b += "<|im_start|>system\n"; b += msgs[0].content; b += "<|im_end|>\n";
    }

    size_t start = has_system ? 1 : 0;
    for (size_t i = start; i < msgs.size(); ++i) {
        const ChatMessage& m = msgs[i];
        if (m.role == "user") {
            b += "<|im_start|>user\n"; b += m.content; b += "<|im_end|>\n";
        } else if (m.role == "assistant") {
            b += "<|im_start|>assistant\n"; b += m.content;
            for (const auto& tc : m.tool_calls) {
                b += "\n<tool_call>\n{\"name\": \"";
                b += tc.name;
                b += "\", \"arguments\": ";
                b += tc.arguments_json.empty() ? "{}" : tc.arguments_json;
                b += "}\n</tool_call>";
            }
            b += "<|im_end|>\n";
        } else if (m.role == "tool") {
            // Merge consecutive tool results into one user turn of <tool_response> blocks.
            if (i == start || msgs[i - 1].role != "tool") b += "<|im_start|>user\n";
            b += "<tool_response>\n"; b += m.content; b += "\n</tool_response>\n";
            if (i + 1 >= msgs.size() || msgs[i + 1].role != "tool") b += "<|im_end|>\n";
        }
    }

    b += "<|im_start|>assistant\n";
    if (!enable_thinking) b += "<think>\n\n</think>\n\n";
    return b;
}
