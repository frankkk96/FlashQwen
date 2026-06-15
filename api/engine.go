package main

import (
	"bufio"
	"encoding/json"
	"net"
)

// engineReq / engineResp are the newline-JSON protocol spoken to the C++ `flashqwen serve`
// process over the Unix socket (see src/server.hpp).
type engineReq struct {
	Prompt      string  `json:"prompt"`
	MaxTokens   int     `json:"max_tokens"`
	Temperature float64 `json:"temperature"`
	TopP        float64 `json:"top_p"`
}

type engineResp struct {
	Delta            string `json:"delta"`
	Done             bool   `json:"done"`
	FinishReason     string `json:"finish_reason"`
	PromptTokens     int    `json:"prompt_tokens"`
	CompletionTokens int    `json:"completion_tokens"`
}

// generate dials the engine, submits one request, and invokes onDelta for every streamed token
// chunk. It returns the terminating done message (finish_reason + token counts). onDelta may be
// nil for the non-streaming path.
func generate(socketPath string, req engineReq, onDelta func(string)) (engineResp, error) {
	conn, err := net.Dial("unix", socketPath)
	if err != nil {
		return engineResp{}, err
	}
	defer conn.Close()

	line, _ := json.Marshal(req)
	line = append(line, '\n')
	if _, err := conn.Write(line); err != nil {
		return engineResp{}, err
	}

	sc := bufio.NewScanner(conn)
	sc.Buffer(make([]byte, 0, 64*1024), 16*1024*1024) // allow long token lines
	for sc.Scan() {
		var r engineResp
		if err := json.Unmarshal(sc.Bytes(), &r); err != nil {
			continue
		}
		if r.Done {
			return r, nil
		}
		if onDelta != nil {
			onDelta(r.Delta)
		}
	}
	if err := sc.Err(); err != nil {
		return engineResp{}, err
	}
	return engineResp{Done: true, FinishReason: "stop"}, nil
}
