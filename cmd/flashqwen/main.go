// flashqwen — the entry point. Owns the model-text layer (tokenizer, ChatML template, tool-call
// detection) and the user-facing modes; embeds and drives the C++ token engine over gRPC.
//
//	flashqwen serve     --model DIR [--addr :8000] [--slots N] [--max-ctx N]
//	flashqwen chat      --model DIR [--max-ctx N]
package main

import (
	"fmt"
	"os"
)

func usage() {
	fmt.Fprint(os.Stderr, `flashqwen — OpenAI-compatible server + CLI for a Qwen3 token engine

usage:
  flashqwen serve     --model DIR [--addr :8000] [--slots N] [--max-ctx N]
  flashqwen chat      --model DIR [--max-ctx N]

All modes embed and launch the C++ engine themselves; no separate process to start.
`)
}

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	switch os.Args[1] {
	case "serve":
		runServe(os.Args[2:])
	case "chat":
		runChat(os.Args[2:])
	case "-h", "--help", "help":
		usage()
	default:
		fmt.Fprintf(os.Stderr, "unknown command %q\n\n", os.Args[1])
		usage()
		os.Exit(2)
	}
}
