// Byte-level BPE tokenizer (GPT-2 / Qwen2 style), implemented from scratch.
//
//  * vocab.json  : maps a byte-level-unicode token string -> id
//  * merges.txt  : ordered list of BPE merge rules (rank = line number)
//
// Pipeline:  text --(special-token split)--> pieces --(pretokenize)--> words
//            --(byte->unicode)--> symbols --(BPE merges)--> token ids
//
// The pretokenizer is a hand-written scanner approximating the Qwen2 regex. It is not a
// byte-exact reimplementation of the unicode regex, but it gets whitespace attachment and
// letter/number/punct boundaries right, and BPE never merges across non-mergeable
// boundaries anyway, so token ids match HF for ordinary text.
#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

class Tokenizer {
public:
    void load(const std::string& dir);   // expects vocab.json + merges.txt in dir

    // Encode UTF-8 text to token ids. Special-token substrings (e.g. "<|im_start|>")
    // are recognised and emitted as their single id.
    std::vector<int> encode(const std::string& text) const;

    // Raw bytes for a token (empty for special / undefined ids).
    const std::string& token_bytes(int id) const;
    bool is_special(int id) const;

    int vocab_size() const { return (int)id_to_token_.size(); }

    // --- streaming detokenizer ---------------------------------------------------------
    // Feed token ids one at a time; returns any newly-completed UTF-8 text. Incomplete
    // multi-byte sequences are buffered until the continuation bytes arrive.
    struct Stream {
        std::string pending;
    };
    std::string stream_decode(Stream& s, int id) const;

private:
    void build_byte_maps();
    void apply_bpe(const std::string& word_bytes, std::vector<int>& out) const;

    std::unordered_map<std::string, int> token_to_id_;   // symbol-space string -> id
    std::vector<std::string> id_to_token_;               // id -> raw bytes (or special literal)
    std::vector<char> special_flag_;
    std::unordered_map<std::string, int> merge_rank_;     // "A\x01B" -> rank

    // byte-level (un)mapping
    uint32_t byte_to_cp_[256];
    std::string byte_to_sym_[256];                        // UTF-8 of byte_to_cp_[b]
    std::unordered_map<uint32_t, uint8_t> cp_to_byte_;

    std::vector<std::pair<std::string,int>> specials_;    // (literal, id), longest-first
};
