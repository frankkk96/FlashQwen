package main

import (
	"context"
	"io"

	pb "flashqwen-api/pb"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// engineClient is the gRPC client to the C++ inference engine.
type engineClient struct {
	conn *grpc.ClientConn
	cli  pb.InferenceClient
}

func dialEngine(addr string) (*engineClient, error) {
	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, err
	}
	return &engineClient{conn: conn, cli: pb.NewInferenceClient(conn)}, nil
}

// generate opens a server-streaming Generate RPC and dispatches each event. It returns the
// terminal Done. Cancelling ctx (e.g. client disconnect) cancels the RPC, which the engine turns
// into a generation abort.
func (e *engineClient) generate(ctx context.Context, req *pb.GenerateRequest,
	onText func(string), onTool func(*pb.ToolCall)) (*pb.Done, error) {
	stream, err := e.cli.Generate(ctx, req)
	if err != nil {
		return nil, err
	}
	for {
		ev, err := stream.Recv()
		if err == io.EOF {
			return &pb.Done{FinishReason: "stop"}, nil
		}
		if err != nil {
			return nil, err
		}
		switch x := ev.Event.(type) {
		case *pb.GenerateEvent_TextDelta:
			if onText != nil {
				onText(x.TextDelta)
			}
		case *pb.GenerateEvent_ToolCall:
			if onTool != nil {
				onTool(x.ToolCall)
			}
		case *pb.GenerateEvent_Done:
			return x.Done, nil
		}
	}
}

func (e *engineClient) modelID(ctx context.Context) (string, error) {
	info, err := e.cli.GetModel(ctx, &pb.ModelRequest{})
	if err != nil {
		return "", err
	}
	return info.Id, nil
}

// toGenerateRequest maps an OpenAI chat request to the engine's structured request. The gateway
// does no model-specific work — the engine renders the template and parses tool calls.
func toGenerateRequest(req ChatRequest) *pb.GenerateRequest {
	g := &pb.GenerateRequest{
		MaxTokens:      int32(req.MaxTokens),
		TopP:           1.0,
		EnableThinking: req.EnableThinking != nil && *req.EnableThinking,
	}
	if req.Temperature != nil {
		g.Temperature = float32(*req.Temperature)
	}
	if req.TopP != nil {
		g.TopP = float32(*req.TopP)
	}
	for _, m := range req.Messages {
		pm := &pb.Message{Role: m.Role, Content: m.Text(), ToolCallId: m.ToolCallID}
		for _, tc := range m.ToolCalls {
			pm.ToolCalls = append(pm.ToolCalls, &pb.ToolCall{
				Id: tc.ID, Name: tc.Function.Name, ArgumentsJson: tc.Function.Arguments})
		}
		g.Messages = append(g.Messages, pm)
	}
	for _, t := range req.Tools {
		g.Tools = append(g.Tools, &pb.ToolDef{
			Name: t.Function.Name, Description: t.Function.Description,
			ParametersJson: string(t.Function.Parameters)})
	}
	return g
}
