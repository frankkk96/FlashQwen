package hub

import (
	"path/filepath"
	"reflect"
	"testing"
)

func TestParseID(t *testing.T) {
	cases := []struct {
		in       string
		wantRepo string
		wantRev  string
	}{
		{"Qwen/Qwen3-8B", "Qwen/Qwen3-8B", "main"},
		{"Qwen/Qwen3-8B@v1.0", "Qwen/Qwen3-8B", "v1.0"},
		{"org/name@abc123", "org/name", "abc123"},
	}
	for _, tc := range cases {
		repo, rev := parseID(tc.in)
		if repo != tc.wantRepo || rev != tc.wantRev {
			t.Errorf("parseID(%q) = (%q, %q), want (%q, %q)", tc.in, repo, rev, tc.wantRepo, tc.wantRev)
		}
	}
}

func TestRepoRe(t *testing.T) {
	valid := []string{"Qwen/Qwen3-8B", "org/name", "a.b/c-d", "models/qwen3-8b"}
	invalid := []string{"justname", "a/b/c", "/leading", "trailing/", "has space/name", ""}
	for _, s := range valid {
		if !repoRe.MatchString(s) {
			t.Errorf("repoRe should match %q", s)
		}
	}
	for _, s := range invalid {
		if repoRe.MatchString(s) {
			t.Errorf("repoRe should not match %q", s)
		}
	}
}

func TestParseShardIndex(t *testing.T) {
	data := []byte(`{
		"metadata": {"total_size": 123},
		"weight_map": {
			"model.embed_tokens.weight": "model-00001-of-00002.safetensors",
			"model.layers.0.mlp.down_proj.weight": "model-00002-of-00002.safetensors",
			"lm_head.weight": "model-00001-of-00002.safetensors"
		}
	}`)
	got, err := parseShardIndex(data)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	want := []string{"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("parseShardIndex = %v, want %v", got, want)
	}

	if _, err := parseShardIndex([]byte(`{"weight_map": {}}`)); err == nil {
		t.Error("expected error on empty weight_map")
	}
	if _, err := parseShardIndex([]byte(`not json`)); err == nil {
		t.Error("expected error on malformed json")
	}
}

func TestCacheRoot(t *testing.T) {
	t.Run("HF_HUB_CACHE wins", func(t *testing.T) {
		t.Setenv("HF_HUB_CACHE", "/tmp/hub-cache")
		t.Setenv("HF_HOME", "/tmp/hf-home")
		if got := cacheRoot(); got != "/tmp/hub-cache" {
			t.Errorf("cacheRoot = %q, want /tmp/hub-cache", got)
		}
	})
	t.Run("HF_HOME falls back to hub subdir", func(t *testing.T) {
		t.Setenv("HF_HUB_CACHE", "")
		t.Setenv("HF_HOME", "/tmp/hf-home")
		if got, want := cacheRoot(), filepath.Join("/tmp/hf-home", "hub"); got != want {
			t.Errorf("cacheRoot = %q, want %q", got, want)
		}
	})
}

func TestEndpoint(t *testing.T) {
	t.Run("default", func(t *testing.T) {
		t.Setenv("HF_ENDPOINT", "")
		if got := endpoint(); got != "https://huggingface.co" {
			t.Errorf("endpoint = %q, want default", got)
		}
	})
	t.Run("mirror with trailing slash trimmed", func(t *testing.T) {
		t.Setenv("HF_ENDPOINT", "https://hf-mirror.com/")
		if got := endpoint(); got != "https://hf-mirror.com" {
			t.Errorf("endpoint = %q, want https://hf-mirror.com", got)
		}
	})
}
