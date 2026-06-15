// Package tokenizer is a from-scratch byte-level BPE tokenizer for Qwen3, loaded from a model
// directory (vocab.json + merges.txt + tokenizer.json's added_tokens). The Go app owns all
// tokenisation; the C++ engine only sees ids.
//
// It is a direct port of the engine's former C++ tokenizer. We do NOT use an off-the-shelf Go
// tokenizer because the only mature pure-Go option (sugarme) can't load Qwen3: it wants the legacy
// string merges format, and — fatally — the GPT-2/Qwen pre-tokenizer regex uses a negative
// lookahead (`\s+(?!\S)`) that Go's RE2 engine cannot compile. So the pre-tokenizer is hand-rolled
// here (matching the regex's intent without lookahead), exactly as the C++ version did.
package tokenizer

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"
)

type special struct {
	str string
	id  int
}

type Tokenizer struct {
	byteToSym [256]string       // GPT-2 byte -> unicode-symbol string
	cpToByte  map[rune]byte     // inverse, for detokenisation
	tokToID   map[string]int    // symbol-space token -> id
	idToTok   []string          // id -> raw bytes (or special literal)
	specialF  []bool            // id -> is special
	mergeRank map[string]int    // "A\x01B" -> rank
	specials  []special         // special literals, longest-first (for encode splitting)
}

// Load reads vocab.json + merges.txt + tokenizer.json (for added/special tokens) from dir.
func Load(dir string) (*Tokenizer, error) {
	t := &Tokenizer{cpToByte: map[rune]byte{}, tokToID: map[string]int{}, mergeRank: map[string]int{}}
	t.buildByteMaps()

	specials, err := loadAddedTokens(dir + "/tokenizer.json")
	if err != nil {
		return nil, err
	}

	vocab, err := loadVocab(dir + "/vocab.json")
	if err != nil {
		return nil, err
	}
	maxID := 0
	for _, id := range vocab {
		if id > maxID {
			maxID = id
		}
	}
	for _, sp := range specials {
		if sp.id > maxID {
			maxID = sp.id
		}
	}
	t.idToTok = make([]string, maxID+1)
	t.specialF = make([]bool, maxID+1)

	for sym, id := range vocab {
		t.tokToID[sym] = id
		t.idToTok[id] = t.symToBytes(sym)
	}
	for _, sp := range specials {
		t.idToTok[sp.id] = sp.str
		t.specialF[sp.id] = true
		t.tokToID[sp.str] = sp.id
		t.specials = append(t.specials, sp)
	}
	sort.Slice(t.specials, func(i, j int) bool { return len(t.specials[i].str) > len(t.specials[j].str) })

	if err := t.loadMerges(dir + "/merges.txt"); err != nil {
		return nil, err
	}
	return t, nil
}

func loadVocab(path string) (map[string]int, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read vocab.json: %w", err)
	}
	var m map[string]int
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, fmt.Errorf("parse vocab.json: %w", err)
	}
	return m, nil
}

func loadAddedTokens(path string) ([]special, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read tokenizer.json: %w", err)
	}
	var doc struct {
		AddedTokens []struct {
			ID      int    `json:"id"`
			Content string `json:"content"`
		} `json:"added_tokens"`
	}
	if err := json.Unmarshal(b, &doc); err != nil {
		return nil, fmt.Errorf("parse tokenizer.json: %w", err)
	}
	out := make([]special, 0, len(doc.AddedTokens))
	for _, a := range doc.AddedTokens {
		out = append(out, special{str: a.Content, id: a.ID})
	}
	return out, nil
}

func (t *Tokenizer) loadMerges(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("open merges.txt: %w", err)
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1024*1024), 1024*1024)
	rank := 0
	for sc.Scan() {
		line := sc.Text()
		if line == "" || line[0] == '#' {
			continue
		}
		sp := strings.IndexByte(line, ' ')
		if sp < 0 {
			continue
		}
		a, b := line[:sp], strings.TrimRight(line[sp+1:], "\r")
		t.mergeRank[a+"\x01"+b] = rank
		rank++
	}
	return sc.Err()
}

// ---- GPT-2 byte<->unicode tables --------------------------------------------------------

func (t *Tokenizer) buildByteMaps() {
	printable := make([]bool, 256)
	add := func(lo, hi int) {
		for b := lo; b <= hi; b++ {
			printable[b] = true
		}
	}
	add('!', '~')
	add(0xA1, 0xAC)
	add(0xAE, 0xFF)
	n := 0
	for b := 0; b < 256; b++ {
		var cp rune
		if printable[b] {
			cp = rune(b)
		} else {
			cp = rune(256 + n)
			n++
		}
		t.byteToSym[b] = string(cp)
		t.cpToByte[cp] = byte(b)
	}
}

// symToBytes decodes a symbol-space string back to its raw bytes.
func (t *Tokenizer) symToBytes(sym string) string {
	var raw strings.Builder
	for _, cp := range sym {
		if b, ok := t.cpToByte[cp]; ok {
			raw.WriteByte(b)
		}
	}
	return raw.String()
}

// ---- public API -------------------------------------------------------------------------

func (t *Tokenizer) Decode(ids []int) string {
	var b strings.Builder
	for _, id := range ids {
		if id >= 0 && id < len(t.idToTok) {
			b.WriteString(t.idToTok[id])
		}
	}
	return b.String()
}

func (t *Tokenizer) ID(token string) (int, bool) {
	id, ok := t.tokToID[token]
	return id, ok
}

func (t *Tokenizer) IsSpecial(id int) bool {
	return id >= 0 && id < len(t.specialF) && t.specialF[id]
}

func (t *Tokenizer) VocabSize() int { return len(t.idToTok) }

// Encode splits text on special-token literals (longest-first, emitting their ids directly) and
// BPE-encodes the text in between.
func (t *Tokenizer) Encode(text string) ([]int, error) {
	var out []int
	pos := 0
	for pos < len(text) {
		best, bestID, bestLen := -1, -1, 0
		for _, sp := range t.specials {
			if f := strings.Index(text[pos:], sp.str); f >= 0 {
				f += pos
				if best < 0 || f < best || (f == best && len(sp.str) > bestLen) {
					best, bestID, bestLen = f, sp.id, len(sp.str)
				}
			}
		}
		chunkEnd := len(text)
		if best >= 0 {
			chunkEnd = best
		}
		if chunkEnd > pos {
			t.pretokenizeAndBPE(text[pos:chunkEnd], &out)
		}
		if best < 0 {
			break
		}
		out = append(out, bestID)
		pos = best + bestLen
	}
	return out, nil
}

// ---- BPE on one pre-token ---------------------------------------------------------------

func (t *Tokenizer) applyBPE(word string, out *[]int) {
	syms := make([]string, 0, len(word))
	for i := 0; i < len(word); i++ {
		syms = append(syms, t.byteToSym[word[i]])
	}
	if len(syms) == 0 {
		return
	}
	for len(syms) >= 2 {
		bestRank, bestI := int(^uint(0)>>1), -1
		for i := 0; i+1 < len(syms); i++ {
			if r, ok := t.mergeRank[syms[i]+"\x01"+syms[i+1]]; ok && r < bestRank {
				bestRank, bestI = r, i
			}
		}
		if bestI < 0 {
			break
		}
		a, b := syms[bestI], syms[bestI+1]
		merged := make([]string, 0, len(syms))
		for i := 0; i < len(syms); {
			if i+1 < len(syms) && syms[i] == a && syms[i+1] == b {
				merged = append(merged, a+b)
				i += 2
			} else {
				merged = append(merged, syms[i])
				i++
			}
		}
		syms = merged
	}
	for _, s := range syms {
		if id, ok := t.tokToID[s]; ok {
			*out = append(*out, id)
		}
	}
}

// ---- pre-tokenizer (hand-rolled GPT-2/Qwen split, no regex) -----------------------------

type cp struct {
	r   rune
	off int // byte offset in text
}

func decodeCPs(s string) []cp {
	out := make([]cp, 0, len(s))
	for i, r := range s {
		out = append(out, cp{r, i})
	}
	return out
}

func isNumber(c rune) bool { return c >= '0' && c <= '9' }
func isSpace(c rune) bool {
	switch c {
	case ' ', '\t', '\n', '\r', '\v', '\f', 0x85, 0xA0, 0x1680, 0x2028, 0x2029, 0x202F, 0x205F, 0x3000, 0xFEFF:
		return true
	}
	return c >= 0x2000 && c <= 0x200A
}
func isNewline(c rune) bool { return c == '\n' || c == '\r' }
func isLetter(c rune) bool {
	if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') {
		return true
	}
	return c >= 0x80 && !isSpace(c) // non-ASCII (CJK, accents, ...) treated as letters
}
func isPunct(c rune) bool { return !isSpace(c) && !isLetter(c) && !isNumber(c) }

func (t *Tokenizer) pretokenizeAndBPE(text string, out *[]int) {
	cps := decodeCPs(text)
	n := len(cps)
	bytesOf := func(a, b int) string {
		bo := cps[a].off
		eo := len(text)
		if b < n {
			eo = cps[b].off
		}
		return text[bo:eo]
	}
	lower := func(c rune) rune {
		if c >= 'A' && c <= 'Z' {
			return c + 32
		}
		return c
	}

	i := 0
	for i < n {
		c := cps[i].r

		// 1) contractions: 's 't 're 've 'm 'll 'd (case-insensitive)
		if c == '\'' && i+1 < n {
			a := lower(cps[i+1].r)
			var b rune
			if i+2 < n {
				b = lower(cps[i+2].r)
			}
			l := 0
			if a == 's' || a == 't' || a == 'm' || a == 'd' {
				l = 2
			} else if (a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l') {
				l = 3
			}
			if l > 0 {
				t.applyBPE(bytesOf(i, i+l), out)
				i += l
				continue
			}
		}

		// 2) optional non-(newline/letter/number) prefix + letters
		if isLetter(c) {
			j := i + 1
			for j < n && isLetter(cps[j].r) {
				j++
			}
			t.applyBPE(bytesOf(i, j), out)
			i = j
			continue
		}
		if !isNewline(c) && !isNumber(c) && i+1 < n && isLetter(cps[i+1].r) {
			j := i + 2
			for j < n && isLetter(cps[j].r) {
				j++
			}
			t.applyBPE(bytesOf(i, j), out)
			i = j
			continue
		}

		// 3) single digit
		if isNumber(c) {
			t.applyBPE(bytesOf(i, i+1), out)
			i++
			continue
		}

		// 4) optional leading space + punctuation run + trailing newlines
		{
			start, k := i, i
			if c == ' ' && i+1 < n && isPunct(cps[i+1].r) {
				k = i + 1
			}
			if isPunct(cps[k].r) {
				j := k
				for j < n && isPunct(cps[j].r) {
					j++
				}
				for j < n && isNewline(cps[j].r) {
					j++
				}
				t.applyBPE(bytesOf(start, j), out)
				i = j
				continue
			}
		}

		// 5) whitespace run (leave one trailing space for the next word, like \s+(?!\S))
		if isSpace(c) {
			j := i
			for j < n && isSpace(cps[j].r) {
				j++
			}
			take := j
			if j < n && !isSpace(cps[j].r) && (j-i) >= 2 {
				take = j - 1
			}
			t.applyBPE(bytesOf(i, take), out)
			i = take
			continue
		}

		// 6) fallback: single codepoint
		t.applyBPE(bytesOf(i, i+1), out)
		i++
	}
}
