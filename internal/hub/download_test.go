package hub

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func sha(b []byte) string { h := sha256.Sum256(b); return hex.EncodeToString(h[:]) }

// fakeHub stands in for a Hugging Face endpoint: every resolve request 302-redirects to a CDN path,
// carrying X-Repo-Commit and X-Linked-Etag (and X-Linked-Size only for "LFS" files, mirroring how
// real small files omit it). The CDN handler serves bytes with Range support.
func fakeHub(t *testing.T, commit string, content map[string][]byte, lfs map[string]bool) (*httptest.Server, *int32) {
	t.Helper()
	var ranged int32
	mux := http.NewServeMux()
	mux.HandleFunc("/org/name/resolve/main/", func(w http.ResponseWriter, r *http.Request) {
		name := strings.TrimPrefix(r.URL.Path, "/org/name/resolve/main/")
		body, ok := content[name]
		if !ok {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		w.Header().Set("X-Repo-Commit", commit)
		w.Header().Set("X-Linked-Etag", `"`+sha(body)+`"`)
		if lfs[name] {
			w.Header().Set("X-Linked-Size", strconv.Itoa(len(body)))
		}
		w.Header().Set("Location", "/cdn/"+name)
		w.WriteHeader(http.StatusFound)
	})
	mux.HandleFunc("/cdn/", func(w http.ResponseWriter, r *http.Request) {
		name := strings.TrimPrefix(r.URL.Path, "/cdn/")
		body, ok := content[name]
		if !ok {
			w.WriteHeader(http.StatusNotFound)
			return
		}
		if r.Header.Get("Range") != "" {
			atomic.StoreInt32(&ranged, 1)
		}
		http.ServeContent(w, r, name, time.Time{}, bytes.NewReader(body))
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv, &ranged
}

func TestDownloadIntegration(t *testing.T) {
	const commit = "b968826d9c46dd6066d109eabc6255188de91218"
	content := map[string][]byte{
		"config.json":                      []byte(`{"architectures":["Qwen3ForCausalLM"]}`),
		"tokenizer.json":                   []byte(`{"tok":1}`),
		"vocab.json":                       []byte(`{"a":0}`),
		"merges.txt":                       []byte("a b\n"),
		"model.safetensors.index.json":     []byte(`{"weight_map":{"x":"model-00001-of-00002.safetensors","y":"model-00002-of-00002.safetensors"}}`),
		"model-00001-of-00002.safetensors": bytes.Repeat([]byte("A"), 5000),
		"model-00002-of-00002.safetensors": bytes.Repeat([]byte("B"), 3000),
		// generation_config.json is intentionally absent → optional 404 → skipped.
	}
	lfs := map[string]bool{
		"model-00001-of-00002.safetensors": true,
		"model-00002-of-00002.safetensors": true,
	}
	srv, _ := fakeHub(t, commit, content, lfs)

	home := t.TempDir()
	t.Setenv("HF_ENDPOINT", srv.URL)
	t.Setenv("HF_HOME", home)
	t.Setenv("HF_HUB_CACHE", "")
	t.Setenv("HF_HUB_OFFLINE", "")

	dir, err := Resolve("org/name")
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}

	// The snapshot directory must be the canonical cache path and carry every fetched file.
	wantSnap := filepath.Join(home, "hub", "models--org--name", "snapshots", commit)
	if dir != wantSnap {
		t.Errorf("snapshot dir = %q, want %q", dir, wantSnap)
	}
	for name, body := range content {
		got, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Fatalf("read %s: %v", name, err)
		}
		if !bytes.Equal(got, body) {
			t.Errorf("%s: content mismatch (%d vs %d bytes)", name, len(got), len(body))
		}
	}
	// Optional file that 404s must not appear.
	if _, err := os.Stat(filepath.Join(dir, "generation_config.json")); err == nil {
		t.Error("generation_config.json should be absent")
	}
	// refs/main records the commit.
	ref, err := os.ReadFile(filepath.Join(home, "hub", "models--org--name", "refs", "main"))
	if err != nil || string(ref) != commit {
		t.Errorf("refs/main = %q (err %v), want %q", ref, err, commit)
	}
	// Snapshot entries are symlinks into blobs/.
	link := filepath.Join(dir, "config.json")
	if fi, err := os.Lstat(link); err != nil || fi.Mode()&os.ModeSymlink == 0 {
		t.Errorf("config.json should be a symlink (mode %v, err %v)", fi.Mode(), err)
	}

	// Second resolve must be a pure cache hit: serve no bytes (point the endpoint at a dead server)
	// and it should still succeed from the blobs already on disk.
	t.Setenv("HF_ENDPOINT", srv.URL) // HEADs still work; blobs already complete so no GET happens
	dir2, err := Resolve("org/name")
	if err != nil {
		t.Fatalf("second Resolve: %v", err)
	}
	if dir2 != dir {
		t.Errorf("second resolve dir = %q, want %q", dir2, dir)
	}
}

func TestFetchResume(t *testing.T) {
	const commit = "cafebabe"
	shard := bytes.Repeat([]byte("Z"), 4096)
	content := map[string][]byte{"model.safetensors": shard}
	lfs := map[string]bool{"model.safetensors": true}
	srv, ranged := fakeHub(t, commit, content, lfs)

	repoDir := t.TempDir()
	c := newClientFor(srv.URL)

	// Seed a partial .part with the first half already on disk.
	etag := sha(shard)
	blobsDir := filepath.Join(repoDir, "blobs")
	if err := os.MkdirAll(blobsDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(blobsDir, etag+".part"), shard[:2048], 0o644); err != nil {
		t.Fatal(err)
	}

	m := meta{name: "model.safetensors", etag: etag, size: int64(len(shard))}
	if err := c.fetch("org/name", "main", repoDir, m, silentProgress()); err != nil {
		t.Fatalf("fetch: %v", err)
	}
	if atomic.LoadInt32(ranged) != 1 {
		t.Error("expected a ranged (resumed) request, got none")
	}
	got, err := os.ReadFile(filepath.Join(blobsDir, etag))
	if err != nil {
		t.Fatalf("read blob: %v", err)
	}
	if !bytes.Equal(got, shard) {
		t.Errorf("resumed blob mismatch (%d bytes)", len(got))
	}
}

// newClientFor builds a client pointed at a test endpoint without touching the process environment.
func newClientFor(endpoint string) *client {
	c := newClient()
	c.endpoint = strings.TrimRight(endpoint, "/")
	return c
}
