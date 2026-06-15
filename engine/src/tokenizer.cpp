#include "tokenizer.hpp"
#include "special_tokens.hpp"
#include "rapidjson/document.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>

// ---------------------------------------------------------------------------------------
// UTF-8 helpers
// ---------------------------------------------------------------------------------------
static void append_utf8(std::string& out, uint32_t cp) {
    if (cp <= 0x7F) out.push_back((char)cp);
    else if (cp <= 0x7FF) {
        out.push_back((char)(0xC0 | (cp >> 6)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        out.push_back((char)(0xE0 | (cp >> 12)));
        out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    } else {
        out.push_back((char)(0xF0 | (cp >> 18)));
        out.push_back((char)(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back((char)(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back((char)(0x80 | (cp & 0x3F)));
    }
}

// Decode UTF-8 string into codepoints, recording the byte offset where each starts.
struct CP { uint32_t cp; size_t byte_off; size_t byte_len; };
static std::vector<CP> decode_utf8(const std::string& s) {
    std::vector<CP> out;
    size_t i = 0, n = s.size();
    while (i < n) {
        uint8_t c = (uint8_t)s[i];
        uint32_t cp; size_t len;
        if (c < 0x80) { cp = c; len = 1; }
        else if ((c >> 5) == 0x6 && i+1 < n) { cp = ((c & 0x1F) << 6) | (s[i+1] & 0x3F); len = 2; }
        else if ((c >> 4) == 0xE && i+2 < n) { cp = ((c & 0x0F) << 12) | ((s[i+1] & 0x3F) << 6) | (s[i+2] & 0x3F); len = 3; }
        else if ((c >> 3) == 0x1E && i+3 < n) { cp = ((c & 0x07) << 18) | ((s[i+1] & 0x3F) << 12) | ((s[i+2] & 0x3F) << 6) | (s[i+3] & 0x3F); len = 4; }
        else { cp = c; len = 1; } // invalid byte: treat as latin1
        out.push_back({cp, i, len});
        i += len;
    }
    return out;
}

// Codepoint classification (approximations; see header note).
static bool is_number(uint32_t c) { return c >= '0' && c <= '9'; }
static bool is_space(uint32_t c) {
    switch (c) {
        case ' ': case '\t': case '\n': case '\r': case '\v': case '\f':
        case 0x85: case 0xA0: case 0x1680: case 0x2028: case 0x2029:
        case 0x202F: case 0x205F: case 0x3000: case 0xFEFF:
            return true;
    }
    return (c >= 0x2000 && c <= 0x200A);
}
static bool is_newline(uint32_t c) { return c == '\n' || c == '\r'; }
static bool is_letter(uint32_t c) {
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) return true;
    return c >= 0x80 && !is_space(c);   // treat non-ASCII (CJK, accents, ...) as letters
}
static bool is_punct(uint32_t c) { return !is_space(c) && !is_letter(c) && !is_number(c); }

// ---------------------------------------------------------------------------------------
// byte <-> unicode tables (GPT-2 bytes_to_unicode)
// ---------------------------------------------------------------------------------------
void Tokenizer::build_byte_maps() {
    std::vector<bool> in_printable(256, false);
    auto add_range = [&](int lo, int hi) { for (int b = lo; b <= hi; ++b) in_printable[b] = true; };
    add_range('!', '~');          // 0x21..0x7E
    add_range(0xA1, 0xAC);
    add_range(0xAE, 0xFF);

    int n = 0;
    for (int b = 0; b < 256; ++b) {
        uint32_t cp;
        if (in_printable[b]) cp = (uint32_t)b;
        else cp = 256 + (n++);
        byte_to_cp_[b] = cp;
        std::string sym; append_utf8(sym, cp);
        byte_to_sym_[b] = sym;
        cp_to_byte_[cp] = (uint8_t)b;
    }
}

// ---------------------------------------------------------------------------------------
// loading
// ---------------------------------------------------------------------------------------
void Tokenizer::load(const std::string& dir) {
    build_byte_maps();

    // The canonical Qwen3 special tokens; ids come from special_tokens.hpp (the single registry).
    static const std::pair<const char*, int> kSpecials[] = {
        {"<|endoftext|>",special::ENDOFTEXT},{"<|im_start|>",special::IM_START},{"<|im_end|>",special::IM_END},
        {"<|object_ref_start|>",special::OBJECT_REF_START},{"<|object_ref_end|>",special::OBJECT_REF_END},
        {"<|box_start|>",special::BOX_START},{"<|box_end|>",special::BOX_END},
        {"<|quad_start|>",special::QUAD_START},{"<|quad_end|>",special::QUAD_END},
        {"<|vision_start|>",special::VISION_START},{"<|vision_end|>",special::VISION_END},
        {"<|vision_pad|>",special::VISION_PAD},{"<|image_pad|>",special::IMAGE_PAD},{"<|video_pad|>",special::VIDEO_PAD},
        {"<tool_call>",special::TOOL_CALL_OPEN},{"</tool_call>",special::TOOL_CALL_CLOSE},
        {"<|fim_prefix|>",special::FIM_PREFIX},{"<|fim_middle|>",special::FIM_MIDDLE},{"<|fim_suffix|>",special::FIM_SUFFIX},
        {"<|fim_pad|>",special::FIM_PAD},{"<|repo_name|>",special::REPO_NAME},{"<|file_sep|>",special::FILE_SEP},
        {"<tool_response>",special::TOOL_RESPONSE_OPEN},{"</tool_response>",special::TOOL_RESPONSE_CLOSE},
        {"<think>",special::THINK_OPEN},{"</think>",special::THINK_CLOSE},
    };

    // vocab.json
    {
        std::ifstream f(dir + "/vocab.json");
        if (!f) throw std::runtime_error("cannot open vocab.json");
        std::stringstream ss; ss << f.rdbuf();
        std::string text = ss.str();
        rapidjson::Document root;
        root.Parse(text.c_str());
        if (root.HasParseError() || !root.IsObject())
            throw std::runtime_error("invalid vocab.json");
        int max_id = 0;
        for (auto& kv : root.GetObject()) max_id = std::max(max_id, kv.value.GetInt());
        for (auto& sp : kSpecials) max_id = std::max(max_id, sp.second);

        id_to_token_.assign(max_id + 1, std::string());
        special_flag_.assign(max_id + 1, 0);

        for (auto& kv : root.GetObject()) {
            const std::string sym = kv.name.GetString();   // byte-level-unicode string
            int id = kv.value.GetInt();
            token_to_id_[sym] = id;
            // decode symbol-space string back to raw bytes for detokenization
            std::string raw;
            for (auto& c : decode_utf8(sym)) {
                auto it = cp_to_byte_.find(c.cp);
                if (it != cp_to_byte_.end()) raw.push_back((char)it->second);
            }
            id_to_token_[id] = raw;
        }
    }

    // special tokens
    for (auto& sp : kSpecials) {
        id_to_token_[sp.second] = sp.first;
        special_flag_[sp.second] = 1;
        token_to_id_[sp.first] = sp.second;          // (not used for BPE, but handy)
        specials_.emplace_back(sp.first, sp.second);
    }
    std::sort(specials_.begin(), specials_.end(),
              [](const auto& a, const auto& b){ return a.first.size() > b.first.size(); });

    // merges.txt
    {
        std::ifstream f(dir + "/merges.txt");
        if (!f) throw std::runtime_error("cannot open merges.txt");
        std::string line; int rank = 0;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            size_t sp = line.find(' ');
            if (sp == std::string::npos) continue;
            std::string a = line.substr(0, sp);
            std::string b = line.substr(sp + 1);
            if (!b.empty() && b.back() == '\r') b.pop_back();
            merge_rank_[a + std::string(1,'\x01') + b] = rank++;
        }
    }
}

bool Tokenizer::is_special(int id) const {
    return id >= 0 && id < (int)special_flag_.size() && special_flag_[id];
}
const std::string& Tokenizer::token_bytes(int id) const {
    static const std::string empty;
    if (id < 0 || id >= (int)id_to_token_.size()) return empty;
    return id_to_token_[id];
}

// ---------------------------------------------------------------------------------------
// BPE on a single pre-token (given as raw bytes)
// ---------------------------------------------------------------------------------------
void Tokenizer::apply_bpe(const std::string& word_bytes, std::vector<int>& out) const {
    // start from one symbol per byte
    std::vector<std::string> syms;
    syms.reserve(word_bytes.size());
    for (unsigned char b : word_bytes) syms.push_back(byte_to_sym_[b]);
    if (syms.empty()) return;

    // iteratively merge the lowest-rank adjacent pair (all occurrences per pass)
    while (syms.size() >= 2) {
        int best_rank = INT32_MAX, best_i = -1;
        for (size_t i = 0; i + 1 < syms.size(); ++i) {
            auto it = merge_rank_.find(syms[i] + std::string(1,'\x01') + syms[i+1]);
            if (it != merge_rank_.end() && it->second < best_rank) {
                best_rank = it->second; best_i = (int)i;
            }
        }
        if (best_i < 0) break;
        // merge the chosen pair-type across the whole sequence
        std::string A = syms[best_i], B = syms[best_i + 1];
        std::vector<std::string> merged;
        merged.reserve(syms.size());
        for (size_t i = 0; i < syms.size(); ) {
            if (i + 1 < syms.size() && syms[i] == A && syms[i+1] == B) {
                merged.push_back(A + B); i += 2;
            } else {
                merged.push_back(syms[i]); i += 1;
            }
        }
        syms.swap(merged);
    }

    for (auto& s : syms) {
        auto it = token_to_id_.find(s);
        if (it != token_to_id_.end()) out.push_back(it->second);
        // (single-byte symbols are always present, so this never silently drops bytes)
    }
}

// ---------------------------------------------------------------------------------------
// pretokenizer + encode
// ---------------------------------------------------------------------------------------
// Encode a piece of text that contains NO special tokens.
static void pretokenize_and_bpe(const Tokenizer& tk,
                                const std::string& text,
                                const std::vector<CP>& cps,
                                std::vector<int>& out,
                                void (Tokenizer::*bpe)(const std::string&, std::vector<int>&) const);

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> out;
    // Split on special-token literals (longest-first), emitting their ids directly.
    size_t pos = 0;
    while (pos < text.size()) {
        size_t best = std::string::npos; int best_id = -1; size_t best_len = 0;
        for (auto& sp : specials_) {
            size_t f = text.find(sp.first, pos);
            if (f != std::string::npos && (best == std::string::npos || f < best ||
                (f == best && sp.first.size() > best_len))) {
                best = f; best_id = sp.second; best_len = sp.first.size();
            }
        }
        size_t chunk_end = (best == std::string::npos) ? text.size() : best;
        if (chunk_end > pos) {
            std::string chunk = text.substr(pos, chunk_end - pos);
            auto cps = decode_utf8(chunk);
            pretokenize_and_bpe(*this, chunk, cps, out, &Tokenizer::apply_bpe);
        }
        if (best == std::string::npos) break;
        out.push_back(best_id);
        pos = best + best_len;
    }
    return out;
}

static void pretokenize_and_bpe(const Tokenizer& tk,
                                const std::string& text,
                                const std::vector<CP>& cps,
                                std::vector<int>& out,
                                void (Tokenizer::*bpe)(const std::string&, std::vector<int>&) const) {
    size_t n = cps.size();
    auto bytes_of = [&](size_t a, size_t b) -> std::string {   // [a,b) in codepoint index
        size_t bo = cps[a].byte_off;
        size_t eo = (b < n) ? cps[b].byte_off : text.size();
        return text.substr(bo, eo - bo);
    };
    auto lower = [](uint32_t c){ return (c>='A'&&c<='Z') ? c+32 : c; };

    size_t i = 0;
    while (i < n) {
        uint32_t c = cps[i].cp;

        // 1) contractions: 's 't 're 've 'm 'll 'd  (case-insensitive)
        if (c == '\'' && i + 1 < n) {
            uint32_t a = lower(cps[i+1].cp);
            uint32_t b = (i + 2 < n) ? lower(cps[i+2].cp) : 0;
            size_t len = 0;
            if (a=='s'||a=='t'||a=='m'||a=='d') len = 2;
            else if ((a=='r'&&b=='e')||(a=='v'&&b=='e')||(a=='l'&&b=='l')) len = 3;
            if (len) { (tk.*bpe)(bytes_of(i, i+len), out); i += len; continue; }
        }

        // 2) optional non-(newline/letter/number) prefix + letters
        if (is_letter(c)) {
            size_t j = i + 1; while (j < n && is_letter(cps[j].cp)) ++j;
            (tk.*bpe)(bytes_of(i, j), out); i = j; continue;
        }
        if (!is_newline(c) && !is_number(c) && i + 1 < n && is_letter(cps[i+1].cp)) {
            size_t j = i + 2; while (j < n && is_letter(cps[j].cp)) ++j;
            (tk.*bpe)(bytes_of(i, j), out); i = j; continue;
        }

        // 3) single digit
        if (is_number(c)) { (tk.*bpe)(bytes_of(i, i+1), out); i += 1; continue; }

        // 4) optional leading space + punctuation run + trailing newlines
        {
            size_t start = i, k = i;
            if (c == ' ' && i + 1 < n && is_punct(cps[i+1].cp)) k = i + 1;  // take the space
            if (is_punct(cps[k].cp)) {
                size_t j = k; while (j < n && is_punct(cps[j].cp)) ++j;
                while (j < n && is_newline(cps[j].cp)) ++j;
                (tk.*bpe)(bytes_of(start, j), out); i = j; continue;
            }
        }

        // 5) whitespace run (leave one trailing space for the next word, like \s+(?!\S))
        if (is_space(c)) {
            size_t j = i; while (j < n && is_space(cps[j].cp)) ++j;
            size_t take = j;
            if (j < n && !is_space(cps[j].cp) && (j - i) >= 2) take = j - 1;
            (tk.*bpe)(bytes_of(i, take), out); i = take; continue;
        }

        // 6) fallback: single codepoint
        (tk.*bpe)(bytes_of(i, i+1), out); i += 1;
    }
}

// ---------------------------------------------------------------------------------------
// streaming detokenizer
// ---------------------------------------------------------------------------------------
static size_t valid_utf8_prefix_len(const std::string& s) {
    size_t i = 0, n = s.size();
    while (i < n) {
        uint8_t c = (uint8_t)s[i];
        size_t len;
        if (c < 0x80) len = 1;
        else if ((c >> 5) == 0x6) len = 2;
        else if ((c >> 4) == 0xE) len = 3;
        else if ((c >> 3) == 0x1E) len = 4;
        else len = 1;                       // invalid lead: emit as-is
        if (i + len > n) break;             // incomplete trailing sequence
        i += len;
    }
    return i;
}

std::string Tokenizer::stream_decode(Stream& s, int id) const {
    if (is_special(id)) return "";          // don't print control tokens
    s.pending += token_bytes(id);
    size_t k = valid_utf8_prefix_len(s.pending);
    std::string emit = s.pending.substr(0, k);
    s.pending.erase(0, k);
    return emit;
}
