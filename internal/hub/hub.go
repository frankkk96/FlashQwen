// Package hub resolves a model argument to a local directory. If the argument is an existing local
// directory it is returned unchanged; otherwise it is treated as a Hugging Face repo id
// (org/name[@revision]) and the files FlashQwen needs are downloaded into the standard Hugging Face
// cache, returning the snapshot directory. Only the token-engine and tokenizer files are fetched —
// config.json, the tokenizer files, and the safetensors shards — never the whole repo.
//
// Behaviour honours the usual environment variables: HF_ENDPOINT (mirror base URL, e.g.
// https://hf-mirror.com), HF_TOKEN (bearer token for gated/private repos), HF_HOME / HF_HUB_CACHE
// (cache location), and HF_HUB_OFFLINE (resolve from cache with no network call).
package hub

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// fileReq is one file to pull from the repo; required files abort the download if missing, optional
// ones are skipped on a 404 (the Go layers fall back when they are absent).
type fileReq struct {
	name     string
	required bool
}

// neededFiles are the non-weight files FlashQwen reads from a model directory: the engine needs
// config.json; the Go tokenizer needs tokenizer.json / vocab.json / merges.txt; the chat layer reads
// generation_config.json (optional, it falls back to canonical Qwen3 eos tokens). The safetensors
// shards are discovered separately from the index.
var neededFiles = []fileReq{
	{"config.json", true},
	{"generation_config.json", false},
	{"tokenizer.json", true},
	{"vocab.json", true},
	{"merges.txt", true},
}

const (
	indexFile    = "model.safetensors.index.json"
	singleWeight = "model.safetensors"
)

// repoRe matches a Hugging Face repo id: exactly one slash, no path traversal. It also matches a
// two-segment local path like "models/qwen3-8b"; that ambiguity is resolved by Resolve checking the
// filesystem first, so only non-existent paths ever reach the network (and 404 with a clear error).
var repoRe = regexp.MustCompile(`^[\w.-]+/[\w.-]+$`)

// Resolve returns a local model directory for arg. An existing directory is returned as-is; anything
// else is parsed as a Hugging Face repo id and downloaded, yielding the cached snapshot directory.
func Resolve(arg string) (string, error) {
	if fi, err := os.Stat(arg); err == nil {
		if fi.IsDir() {
			return arg, nil
		}
		return "", fmt.Errorf("hub: %q is a file; expected a model directory or a Hugging Face repo id", arg)
	}
	repo, rev := parseID(arg)
	if !repoRe.MatchString(repo) {
		return "", fmt.Errorf("hub: %q is neither an existing directory nor a valid Hugging Face repo id (org/name)", arg)
	}
	return download(repo, rev)
}

// parseID splits "org/name@revision" into the repo id and revision (default "main").
func parseID(arg string) (repo, rev string) {
	rev = "main"
	if i := strings.LastIndex(arg, "@"); i >= 0 {
		return arg[:i], arg[i+1:]
	}
	return arg, rev
}

// download fetches every file FlashQwen needs for repo@rev into the Hugging Face cache and returns
// the assembled snapshot directory.
func download(repo, rev string) (string, error) {
	repoDir := filepath.Join(cacheRoot(), "models--"+strings.ReplaceAll(repo, "/", "--"))
	if offline() {
		return offlineSnapshot(repoDir, rev)
	}
	c := newClient()

	files := append([]fileReq(nil), neededFiles...)

	// Discover the weights: prefer the shard index, fall back to a single safetensors file. The
	// index is small, so fetch it up front to enumerate the shards before the big download.
	idxMeta, commit, found, err := c.head(repo, rev, indexFile)
	if err != nil {
		return "", err
	}
	if found {
		files = append(files, fileReq{indexFile, true})
		if err := c.fetch(repo, rev, repoDir, idxMeta, silentProgress()); err != nil {
			return "", err
		}
		data, err := os.ReadFile(filepath.Join(repoDir, "blobs", idxMeta.etag))
		if err != nil {
			return "", err
		}
		shards, err := parseShardIndex(data)
		if err != nil {
			return "", err
		}
		for _, s := range shards {
			files = append(files, fileReq{s, true})
		}
	} else {
		files = append(files, fileReq{singleWeight, true})
	}

	metas, headCommit, err := c.headAll(repo, rev, files)
	if err != nil {
		return "", err
	}
	if commit == "" {
		commit = headCommit
	}

	if err := c.fetchAll(repo, rev, repoDir, metas); err != nil {
		return "", err
	}
	return buildSnapshot(repoDir, rev, commit, metas)
}

// parseShardIndex returns the unique, sorted set of shard filenames named in a safetensors index's
// weight_map.
func parseShardIndex(data []byte) ([]string, error) {
	var idx struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if err := json.Unmarshal(data, &idx); err != nil {
		return nil, fmt.Errorf("hub: parse %s: %w", indexFile, err)
	}
	set := make(map[string]struct{}, len(idx.WeightMap))
	for _, shard := range idx.WeightMap {
		set[shard] = struct{}{}
	}
	if len(set) == 0 {
		return nil, fmt.Errorf("hub: %s has no weight_map entries", indexFile)
	}
	out := make([]string, 0, len(set))
	for s := range set {
		out = append(out, s)
	}
	sort.Strings(out)
	return out, nil
}
