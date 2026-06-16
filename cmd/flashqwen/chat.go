package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"flashqwen/internal/chatml"
	"flashqwen/internal/engine"
)

func runChat(args []string) {
	fs := flag.NewFlagSet("chat", flag.ExitOnError)
	model := fs.String("model", "", "model directory or Hugging Face repo id (required)")
	maxCtx := fs.Int("max-ctx", 4096, "KV / context length")
	think := fs.Bool("think", false, "enable Qwen3 thinking mode")
	fs.Parse(args)
	if *model == "" {
		log.Fatal("chat: --model is required")
	}

	s, err := open(*model, 1, *maxCtx)
	if err != nil {
		log.Fatalf("chat: %v", err)
	}
	defer s.stop()
	thinking := *think

	fmt.Println("\nFlashQwen interactive chat. Commands: /exit /quit /reset /think on|off")
	var history []chatml.Message
	in := bufio.NewReader(os.Stdin)
	for {
		fmt.Print("\n\033[1mYou:\033[0m ")
		line, err := in.ReadString('\n')
		if err != nil {
			fmt.Println()
			return // EOF
		}
		line = strings.TrimRight(line, "\r\n")
		switch line {
		case "/exit", "/quit":
			return
		case "":
			continue
		case "/reset":
			history = nil
			fmt.Println("[context cleared]")
			continue
		case "/think on":
			thinking = true
			fmt.Println("[thinking on]")
			continue
		case "/think off":
			thinking = false
			fmt.Println("[thinking off]")
			continue
		}

		history = append(history, chatml.Message{Role: "user", Content: line})
		fmt.Print("\033[1mAssistant:\033[0m ")
		res, err := s.eng.Generate(context.Background(), engine.Request{
			Messages: history, EnableThinking: thinking, // MaxTokens 0: fill the remaining context
		}, func(text string, _ *chatml.ToolCall) {
			fmt.Print(text)
		})
		fmt.Println()
		if err != nil {
			fmt.Printf("[error: %v]\n", err)
			history = history[:len(history)-1]
			continue
		}
		history = append(history, chatml.Message{Role: "assistant", Content: res.Text})
	}
}
