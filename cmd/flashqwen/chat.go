package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"flashqwen/internal/chat"
	pb "flashqwen/internal/enginepb"
)

func runChat(args []string) {
	fs := flag.NewFlagSet("chat", flag.ExitOnError)
	model := fs.String("model", "", "model directory (required)")
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
	var history []chat.Message
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

		history = append(history, chat.Message{Role: "user", Content: line})
		prompt := s.cm.Render(history, nil, thinking)
		ids, err := s.tok.Encode(prompt)
		if err != nil {
			fmt.Printf("[tokenise error: %v]\n", err)
			history = history[:len(history)-1]
			continue
		}
		room := s.maxCtx() - len(ids)
		if room < 1 {
			fmt.Println("[context full - type /reset to clear history]")
			history = history[:len(history)-1]
			continue
		}
		maxNew := room
		if maxNew > 1024 {
			maxNew = 1024
		}
		in32 := make([]int32, len(ids))
		for i, id := range ids {
			in32[i] = int32(id)
		}

		fmt.Print("\033[1mAssistant:\033[0m ")
		stream := s.cm.NewStream()
		var sb strings.Builder
		_, err = s.eng.Generate(context.Background(), &pb.GenerateRequest{
			InputIds: in32, MaxTokens: int32(maxNew), TopP: 1.0, StopTokenIds: s.cm.StopTokenIDs(),
		}, func(id int32) {
			if text, _ := stream.Push(int(id)); text != "" {
				fmt.Print(text)
				sb.WriteString(text)
			}
		})
		fmt.Println()
		if err != nil {
			fmt.Printf("[engine error: %v]\n", err)
			history = history[:len(history)-1]
			continue
		}
		history = append(history, chat.Message{Role: "assistant", Content: sb.String()})
	}
}
