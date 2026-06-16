package hub

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// cacheRoot is where Hugging Face caches live: HF_HUB_CACHE wins, else $HF_HOME/hub, else the
// canonical ~/.cache/huggingface/hub. This is the same resolution huggingface_hub uses, so a model
// already pulled by huggingface-cli or Python is reused rather than re-downloaded.
func cacheRoot() string {
	if v := os.Getenv("HF_HUB_CACHE"); v != "" {
		return v
	}
	if v := os.Getenv("HF_HOME"); v != "" {
		return filepath.Join(v, "hub")
	}
	home, err := os.UserHomeDir()
	if err != nil {
		home = "."
	}
	return filepath.Join(home, ".cache", "huggingface", "hub")
}

// offline reports whether HF_HUB_OFFLINE asks us to resolve purely from cache.
func offline() bool {
	switch strings.ToLower(os.Getenv("HF_HUB_OFFLINE")) {
	case "1", "true", "yes", "on":
		return true
	}
	return false
}

// buildSnapshot lays out the snapshots/<commit>/ directory as symlinks into blobs/, and records
// refs/<rev> = commit, matching the standard Hugging Face cache layout. It returns the snapshot
// directory, which is what the engine and tokenizer are pointed at.
func buildSnapshot(repoDir, rev, commit string, metas map[string]meta) (string, error) {
	snap := filepath.Join(repoDir, "snapshots", commit)
	if err := os.MkdirAll(snap, 0o755); err != nil {
		return "", err
	}
	for name, m := range metas {
		link := filepath.Join(snap, name)
		blob := filepath.Join(repoDir, "blobs", m.etag)
		if err := os.MkdirAll(filepath.Dir(link), 0o755); err != nil {
			return "", err
		}
		if _, err := os.Lstat(link); err == nil {
			os.Remove(link) // replace any stale link from an earlier revision
		}
		// Relative symlink so the cache stays movable; fall back to a copy where symlinks are not
		// supported (e.g. some Windows or network filesystems).
		rel, err := filepath.Rel(filepath.Dir(link), blob)
		if err != nil {
			rel = blob
		}
		if err := os.Symlink(rel, link); err != nil {
			if err := copyFile(blob, link); err != nil {
				return "", err
			}
		}
	}
	ref := filepath.Join(repoDir, "refs", rev)
	if err := os.MkdirAll(filepath.Dir(ref), 0o755); err != nil {
		return "", err
	}
	if err := os.WriteFile(ref, []byte(commit), 0o644); err != nil {
		return "", err
	}
	return snap, nil
}

// offlineSnapshot resolves repo@rev from cache without any network call. rev may be a ref recorded
// under refs/, or a commit sha that already has a snapshot directory.
func offlineSnapshot(repoDir, rev string) (string, error) {
	name := filepath.Base(repoDir)
	if commit, err := os.ReadFile(filepath.Join(repoDir, "refs", rev)); err == nil {
		snap := filepath.Join(repoDir, "snapshots", strings.TrimSpace(string(commit)))
		if fi, err := os.Stat(snap); err == nil && fi.IsDir() {
			return snap, nil
		}
	}
	snap := filepath.Join(repoDir, "snapshots", rev)
	if fi, err := os.Stat(snap); err == nil && fi.IsDir() {
		return snap, nil
	}
	return "", fmt.Errorf("hub: offline and %s@%s is not in cache (%s)", name, rev, repoDir)
}

func copyFile(src, dst string) error {
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		return err
	}
	return out.Close()
}
