package bench

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// record holds the client-side measurements for one request (or its failure).
type record struct {
	promptTokens  int             // authoritative, from the trailing usage chunk
	outputTokens  int             // authoritative, from the trailing usage chunk
	ttft          time.Duration   // send -> first content chunk
	e2el          time.Duration   // send -> last content chunk
	itls          []time.Duration // gaps between consecutive content chunks
	contentChunks int
	err           error
}

// ---- wire types (the subset the bench sends / reads) -------------------------------------------

type chatReq struct {
	Model         string        `json:"model"`
	Messages      []chatMsg     `json:"messages"`
	MaxTokens     int           `json:"max_tokens"`
	Temperature   float64       `json:"temperature"`
	TopP          float64       `json:"top_p"`
	Stream        bool          `json:"stream"`
	StreamOptions streamOptions `json:"stream_options"`
}

type chatMsg struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type streamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type sseChunk struct {
	Choices []struct {
		Delta struct {
			Content *string `json:"content"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage"`
	Error *struct {
		Message string `json:"message"`
		Type    string `json:"type"`
	} `json:"error"`
}

// streamOnce fires one streaming chat completion and times the token stream. TTFT is the gap to the
// first content delta; ITLs are the gaps between consecutive content deltas; E2EL runs to the last.
// Token counts come from the trailing usage chunk (stream_options.include_usage). A delta may carry
// more than one token (the server flushes on UTF-8 rune boundaries), so ITL is strictly an
// inter-chunk latency — a close proxy for inter-token latency, not an exact per-token figure.
func streamOnce(ctx context.Context, hc *http.Client, baseURL, model string, spec reqSpec, temp, topP float64) record {
	body, _ := json.Marshal(chatReq{
		Model:         model,
		Messages:      []chatMsg{{Role: "user", Content: spec.prompt}},
		MaxTokens:     spec.maxTokens,
		Temperature:   temp,
		TopP:          topP,
		Stream:        true,
		StreamOptions: streamOptions{IncludeUsage: true},
	})
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, baseURL+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return record{err: err}
	}
	httpReq.Header.Set("Content-Type", "application/json")

	st := time.Now()
	resp, err := hc.Do(httpReq)
	if err != nil {
		return record{err: err}
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 2<<10))
		return record{err: fmt.Errorf("http %d: %s", resp.StatusCode, strings.TrimSpace(string(msg)))}
	}

	var rec record
	var last time.Time
	br := bufio.NewReader(resp.Body)
	for {
		line, rerr := br.ReadString('\n')
		if data, ok := strings.CutPrefix(strings.TrimRight(line, "\r\n"), "data: "); ok {
			if data == "[DONE]" {
				break
			}
			var ch sseChunk
			if json.Unmarshal([]byte(data), &ch) == nil {
				if ch.Error != nil {
					rec.err = fmt.Errorf("%s", ch.Error.Message)
				}
				if ch.Usage != nil {
					rec.promptTokens = ch.Usage.PromptTokens
					rec.outputTokens = ch.Usage.CompletionTokens
				}
				if len(ch.Choices) > 0 {
					if ct := ch.Choices[0].Delta.Content; ct != nil && *ct != "" {
						now := time.Now()
						if rec.contentChunks == 0 {
							rec.ttft = now.Sub(st)
						} else {
							rec.itls = append(rec.itls, now.Sub(last))
						}
						last = now
						rec.contentChunks++
					}
				}
			}
		}
		if rerr != nil {
			break // EOF (normal end) or a read error
		}
	}
	if !last.IsZero() {
		rec.e2el = last.Sub(st)
	}
	if rec.err == nil {
		if ce := ctx.Err(); ce != nil {
			rec.err = ce
		} else if rec.contentChunks == 0 && rec.outputTokens == 0 {
			rec.err = errors.New("no tokens received")
		}
	}
	return rec
}
