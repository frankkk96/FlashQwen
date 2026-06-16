package hub

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// meta is the per-file metadata read from a resolve response: the content etag (blob filename) and
// the content size. The commit sha is the same across a request and is carried separately.
type meta struct {
	name string
	etag string
	size int64
}

// client talks to a Hugging Face endpoint. It keeps two http.Clients: meta does HEADs with redirects
// disabled (so the resolve endpoint's own headers — X-Repo-Commit, X-Linked-Etag — are read instead
// of the CDN's), and body follows redirects to stream file contents from the LFS CDN.
type client struct {
	endpoint string
	token    string
	meta     *http.Client
	body     *http.Client
}

func newClient() *client {
	return &client{
		endpoint: endpoint(),
		token:    hubToken(),
		meta: &http.Client{
			CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
			Timeout:       30 * time.Second,
		},
		body: &http.Client{}, // no timeout: weight shards are large
	}
}

func endpoint() string {
	if v := strings.TrimRight(os.Getenv("HF_ENDPOINT"), "/"); v != "" {
		return v
	}
	return "https://huggingface.co"
}

func hubToken() string {
	if v := os.Getenv("HF_TOKEN"); v != "" {
		return v
	}
	return os.Getenv("HUGGING_FACE_HUB_TOKEN")
}

func (c *client) resolveURL(repo, rev, name string) string {
	return fmt.Sprintf("%s/%s/resolve/%s/%s", c.endpoint, repo, rev, name)
}

func (c *client) authorize(r *http.Request) {
	if c.token != "" {
		r.Header.Set("Authorization", "Bearer "+c.token)
	}
}

// head reads one file's metadata without downloading it. found is false on a 404.
func (c *client) head(repo, rev, name string) (m meta, commit string, found bool, err error) {
	req, err := http.NewRequest(http.MethodHead, c.resolveURL(repo, rev, name), nil)
	if err != nil {
		return meta{}, "", false, err
	}
	c.authorize(req)
	resp, err := c.meta.Do(req)
	if err != nil {
		return meta{}, "", false, fmt.Errorf("hub: HEAD %s: %w", name, err)
	}
	defer resp.Body.Close()

	switch {
	case resp.StatusCode == http.StatusNotFound:
		return meta{}, "", false, nil
	case resp.StatusCode == http.StatusUnauthorized, resp.StatusCode == http.StatusForbidden:
		return meta{}, "", false, fmt.Errorf("hub: %s/%s: access denied (set HF_TOKEN for gated or private repos)", repo, name)
	case resp.StatusCode >= 400:
		return meta{}, "", false, fmt.Errorf("hub: HEAD %s: %s", name, resp.Status)
	}
	// 200 (small file served directly) or 3xx (LFS redirect) — both carry the headers we need.
	commit = resp.Header.Get("X-Repo-Commit")
	etag := strings.Trim(firstNonEmpty(resp.Header.Get("X-Linked-Etag"), resp.Header.Get("ETag")), `"`)
	if etag == "" {
		return meta{}, "", false, fmt.Errorf("hub: %s: response had no ETag", name)
	}
	// X-Linked-Size is the authoritative content length. Content-Length is only trustworthy on a
	// direct 200; on a redirect it measures the redirect body, not the file. When neither applies
	// (small files served via a resolve-cache redirect) the size is left unknown (0) and verified
	// loosely — the etag, a content hash, still pins correctness.
	size := headerInt(resp.Header.Get("X-Linked-Size"))
	if size == 0 && resp.StatusCode == http.StatusOK {
		size = headerInt(resp.Header.Get("Content-Length"))
	}
	return meta{name: name, etag: etag, size: size}, commit, true, nil
}

// headAll heads every file, dropping optional 404s and erroring on missing required files. It also
// returns the repo commit (shared by all files in the request).
func (c *client) headAll(repo, rev string, files []fileReq) (map[string]meta, string, error) {
	metas := make(map[string]meta, len(files))
	var commit string
	for _, f := range files {
		m, cm, found, err := c.head(repo, rev, f.name)
		if err != nil {
			return nil, "", err
		}
		if !found {
			if f.required {
				return nil, "", fmt.Errorf("hub: %s/%s not found on %s (revision %q)", repo, f.name, c.endpoint, rev)
			}
			continue
		}
		if cm != "" {
			commit = cm
		}
		metas[f.name] = m
	}
	if commit == "" {
		return nil, "", fmt.Errorf("hub: could not determine commit for %s@%s", repo, rev)
	}
	return metas, commit, nil
}

// fetch downloads one file into blobs/<etag>, resuming a partial .part file with a Range request and
// skipping the download entirely when the blob already exists at the right size.
func (c *client) fetch(repo, rev, repoDir string, m meta, prog *progress) error {
	blob := filepath.Join(repoDir, "blobs", m.etag)
	// A blob is named by its content hash, so an existing one at the expected size is reusable. When
	// the size is unknown (m.size == 0) an existing blob is trusted on the etag alone.
	if fi, err := os.Stat(blob); err == nil && (fi.Size() == m.size || m.size == 0) {
		prog.add(fi.Size()) // already cached (a prior run, or huggingface_hub)
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(blob), 0o755); err != nil {
		return err
	}
	part := blob + ".part"

	var have int64
	if fi, err := os.Stat(part); err == nil && fi.Size() <= m.size {
		have = fi.Size()
	}

	req, err := http.NewRequest(http.MethodGet, c.resolveURL(repo, rev, m.name), nil)
	if err != nil {
		return err
	}
	c.authorize(req)
	if have > 0 {
		req.Header.Set("Range", "bytes="+strconv.FormatInt(have, 10)+"-")
	}
	resp, err := c.body.Do(req)
	if err != nil {
		return fmt.Errorf("hub: GET %s: %w", m.name, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		return fmt.Errorf("hub: GET %s: %s", m.name, resp.Status)
	}

	flags := os.O_CREATE | os.O_WRONLY
	if have > 0 && resp.StatusCode == http.StatusPartialContent {
		flags |= os.O_APPEND
		prog.add(have) // count the bytes already on disk
	} else {
		have = 0 // server ignored Range, or a fresh download: start over
		flags |= os.O_TRUNC
	}
	f, err := os.OpenFile(part, flags, 0o644)
	if err != nil {
		return err
	}
	if _, err := io.Copy(io.MultiWriter(f, prog), resp.Body); err != nil {
		f.Close()
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	if m.size > 0 {
		if fi, err := os.Stat(part); err != nil {
			return err
		} else if fi.Size() != m.size {
			return fmt.Errorf("hub: %s: got %d bytes, expected %d", m.name, fi.Size(), m.size)
		}
	}
	return os.Rename(part, blob)
}

// fetchAll downloads every file concurrently (bounded), reporting aggregate progress.
func (c *client) fetchAll(repo, rev, repoDir string, metas map[string]meta) error {
	var total int64
	for _, m := range metas {
		total += m.size
	}
	prog := newProgress(repo, total)
	defer prog.done()

	const concurrency = 4
	sem := make(chan struct{}, concurrency)
	var wg sync.WaitGroup
	var mu sync.Mutex
	var firstErr error

	for _, m := range metas {
		sem <- struct{}{}
		wg.Add(1)
		go func(m meta) {
			defer wg.Done()
			defer func() { <-sem }()
			if err := c.fetch(repo, rev, repoDir, m, prog); err != nil {
				mu.Lock()
				if firstErr == nil {
					firstErr = err
				}
				mu.Unlock()
			}
		}(m)
	}
	wg.Wait()
	return firstErr
}

// progress prints a single aggregate download line to stderr, updated on a ticker so concurrent
// downloads do not interleave their output.
type progress struct {
	repo  string
	total int64
	got   atomic.Int64
	stop  chan struct{}
	wg    sync.WaitGroup
}

func newProgress(repo string, total int64) *progress {
	p := &progress{repo: repo, total: total, stop: make(chan struct{})}
	if total > 0 && repo != "" {
		p.wg.Add(1)
		go p.loop()
	}
	return p
}

func silentProgress() *progress { return newProgress("", 0) }

func (p *progress) Write(b []byte) (int, error) {
	p.got.Add(int64(len(b)))
	return len(b), nil
}

func (p *progress) add(n int64) { p.got.Add(n) }

func (p *progress) loop() {
	defer p.wg.Done()
	t := time.NewTicker(500 * time.Millisecond)
	defer t.Stop()
	for {
		select {
		case <-p.stop:
			p.print()
			fmt.Fprintln(os.Stderr)
			return
		case <-t.C:
			p.print()
		}
	}
}

func (p *progress) print() {
	d := p.got.Load()
	pct := 100.0
	if p.total > 0 {
		pct = float64(d) / float64(p.total) * 100
	}
	if pct > 100 { // small files of unknown size aren't in the total; keep the bar sane
		pct = 100
	}
	fmt.Fprintf(os.Stderr, "\rdownloading %s: %s / %s (%.0f%%)      ", p.repo, human(d), human(p.total), pct)
}

// done stops the ticker (if running) and prints the final line.
func (p *progress) done() {
	if p.total > 0 && p.repo != "" {
		close(p.stop)
		p.wg.Wait()
	}
}

func human(n int64) string {
	const gb, mb = 1 << 30, 1 << 20
	if n >= gb {
		return fmt.Sprintf("%.2f GB", float64(n)/gb)
	}
	return fmt.Sprintf("%.0f MB", float64(n)/mb)
}

func firstNonEmpty(vs ...string) string {
	for _, v := range vs {
		if v != "" {
			return v
		}
	}
	return ""
}

func headerInt(s string) int64 {
	n, _ := strconv.ParseInt(s, 10, 64)
	return n
}
