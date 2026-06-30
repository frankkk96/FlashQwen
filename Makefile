# FlashQwen build: the Go binary embeds the compiled C++ engine, so the order is
#   engine (cmake) -> copy into the embed dir -> go build.
#
#   make            # build everything -> ./flashqwen
#   make engine     # build just the C++ token engine
#   make proto      # regenerate Go gRPC stubs (only needed when proto/engine.proto changes)
#   make test       # go tests (tokenizer parity needs FLASHQWEN_MODEL set)
#   make run        # build + serve (MODEL=<dir>)

MODEL      ?= ./models/qwen3-8b
ENGINE_BIN := internal/supervisor/bin/flashqwen-engine
GOBIN      := $(shell go env GOPATH)/bin

.PHONY: all go embed engine proto test run clean

all: go

go: embed
	go build -o flashqwen ./cmd/flashqwen

embed: engine
	@mkdir -p $(dir $(ENGINE_BIN))
	cp engine/build/flashqwen-engine $(ENGINE_BIN)

engine:
	cmake -S engine -B engine/build -DCMAKE_BUILD_TYPE=Release
	cmake --build engine/build -j

# C++ proto stubs are generated inside the cmake build; this regenerates the Go stubs.
proto:
	PATH="$(PATH):$(GOBIN)" protoc -I proto \
		--go_out=. --go_opt=module=flashqwen \
		--go-grpc_out=. --go-grpc_opt=module=flashqwen \
		proto/engine.proto

test:
	go test ./...

run: go
	./flashqwen serve --model $(MODEL)

clean:
	rm -rf engine/build flashqwen $(ENGINE_BIN)
