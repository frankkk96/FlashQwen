#include "chat.hpp"
#include "generate.hpp"
#include "prompt.hpp"
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

// Build the ChatML text to append for one user turn (incremental: only the new turn is encoded,
// since the KV cache persists across turns). For turns after the first, a leading <|im_end|>
// closes the previous, streamed assistant reply. The turn/header markers come from prompt.* so
// the format lives in one place.
static std::string user_chunk(const std::string& msg, bool first, bool think) {
    std::string s;
    if (!first) s += "<|im_end|>\n";        // close the previous (streamed) assistant reply
    append_user_turn(s, msg);
    append_assistant_header(s, think);
    return s;
}

int run_chat(Model& model, const Tokenizer& tok, SampleParams sp, std::mt19937& rng,
             bool think, int max_ctx) {
    std::printf("\nFlashQwen interactive chat. Commands: /exit /quit /reset /think on|off\n");
    int past = 0; bool first = true;
    while (true) {
        std::printf("\n\033[1mYou:\033[0m ");
        std::fflush(stdout);
        std::string line;
        if (!std::getline(std::cin, line)) { std::printf("\n"); break; }   // EOF
        if (line == "/exit" || line == "/quit") break;
        if (line.empty()) continue;
        if (line == "/reset") { past = 0; first = true; std::printf("[context cleared]\n"); continue; }
        if (line == "/think on")  { think = true;  std::printf("[thinking on]\n");  continue; }
        if (line == "/think off") { think = false; std::printf("[thinking off]\n"); continue; }

        std::vector<int> ids = tok.encode(user_chunk(line, first, think));
        if (past + (int)ids.size() + 64 >= max_ctx) {
            std::printf("[context full - type /reset to clear history]\n");
            continue;
        }
        first = false;

        std::printf("\033[1mAssistant:\033[0m ");
        std::fflush(stdout);
        generate(model, tok, ids, past, max_ctx, sp, rng, /*stream=*/true, /*stop_on_eos=*/true);
        std::printf("\n");
    }
    return 0;
}
