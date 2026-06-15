// The one place Qwen3 special-token ids are written. Everything that needs an id (the tokenizer's
// string->id table, EOS checks in the decode loops, tool-call detection in the server) references
// these named constants, so no magic number is duplicated. Ids are stable across the Qwen3 family.
#pragma once

namespace special {
constexpr int ENDOFTEXT        = 151643;
constexpr int IM_START         = 151644;
constexpr int IM_END           = 151645;
constexpr int OBJECT_REF_START = 151646;
constexpr int OBJECT_REF_END   = 151647;
constexpr int BOX_START        = 151648;
constexpr int BOX_END          = 151649;
constexpr int QUAD_START       = 151650;
constexpr int QUAD_END         = 151651;
constexpr int VISION_START     = 151652;
constexpr int VISION_END       = 151653;
constexpr int VISION_PAD       = 151654;
constexpr int IMAGE_PAD        = 151655;
constexpr int VIDEO_PAD        = 151656;
constexpr int TOOL_CALL_OPEN   = 151657;  // <tool_call>
constexpr int TOOL_CALL_CLOSE  = 151658;  // </tool_call>
constexpr int FIM_PREFIX       = 151659;
constexpr int FIM_MIDDLE       = 151660;
constexpr int FIM_SUFFIX       = 151661;
constexpr int FIM_PAD          = 151662;
constexpr int REPO_NAME        = 151663;
constexpr int FILE_SEP         = 151664;
constexpr int TOOL_RESPONSE_OPEN  = 151665;  // <tool_response>
constexpr int TOOL_RESPONSE_CLOSE = 151666;  // </tool_response>
constexpr int THINK_OPEN       = 151667;  // <think>
constexpr int THINK_CLOSE      = 151668;  // </think>

// A generation stops on either end-of-turn marker.
inline bool is_eos(int id) { return id == IM_END || id == ENDOFTEXT; }
}  // namespace special
