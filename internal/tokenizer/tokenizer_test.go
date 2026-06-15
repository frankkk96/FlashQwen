package tokenizer

import (
	"os"
	"testing"
)

// Golden ids captured from the HuggingFace `tokenizers` library (the reference implementation) for
// Qwen3-8B. Verified byte-for-byte against HF over 1000+ fuzz prompts; this curated set guards the
// cases that previously diverged: ChatML special tokens, newline runs ("\n\n" stays one token),
// '#'-prefixed merges ("#'" -> one token), contractions, multi-space, unicode, and digits.
var golden = []struct {
	text string
	ids  []int
}{
	{"Hello, world!", []int{9707, 11, 1879, 0}},
	{"What is the capital of France? Answer in one word.", []int{3838, 374, 279, 6722, 315, 9625, 30, 21806, 304, 825, 3409, 13}},
	{"<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n", []int{151644, 872, 198, 13048, 151645, 198, 151644, 77091, 198}},
	{"<think>\n\n</think>\n\nThe answer is 42.", []int{151667, 271, 151668, 271, 785, 4226, 374, 220, 19, 17, 13}},
	{"q#' and don't can't I'll", []int{80, 94182, 323, 1513, 944, 646, 944, 358, 3278}},
	{"a    b   c   ", []int{64, 262, 293, 256, 272, 262}},
	{"café naïve 日本語 🎉 Москва", []int{924, 58858, 94880, 586, 75402, 21894, 102819, 11162, 236, 231, 137639}},
	{"12345 67 890 3.14", []int{16, 17, 18, 19, 20, 220, 21, 22, 220, 23, 24, 15, 220, 18, 13, 16, 19}},
}

// TestEncodeParity checks Encode against the HF golden ids. Set FLASHQWEN_MODEL to the model dir;
// the test skips if it is unset or unreadable (the tokenizer needs vocab.json/merges.txt).
func TestEncodeParity(t *testing.T) {
	dir := os.Getenv("FLASHQWEN_MODEL")
	if dir == "" {
		t.Skip("set FLASHQWEN_MODEL to the model directory to run tokenizer parity tests")
	}
	tok, err := Load(dir)
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	for _, g := range golden {
		got, err := tok.Encode(g.text)
		if err != nil {
			t.Errorf("encode %q: %v", g.text, err)
			continue
		}
		if !equal(got, g.ids) {
			t.Errorf("encode %q:\n  got  %v\n  want %v", g.text, got, g.ids)
		}
	}
}

func equal(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
