// Package engine is the gRPC client to the C++ token engine (flashqwen-engine): prompt token ids
// in, sampled token ids out.
package engine

import (
	"context"
	"io"

	pb "flashqwen/internal/enginepb"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

type Client struct {
	conn *grpc.ClientConn
	cli  pb.EngineClient
}

func Dial(addr string) (*Client, error) {
	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return nil, err
	}
	return &Client{conn: conn, cli: pb.NewEngineClient(conn)}, nil
}

func (c *Client) Close() error { return c.conn.Close() }

// Generate opens a server-streaming Generate RPC, invoking onToken for each produced id, and
// returns the terminal Done. Cancelling ctx (client disconnect) cancels the RPC, which the engine
// turns into a generation abort.
func (c *Client) Generate(ctx context.Context, req *pb.GenerateRequest, onToken func(int32)) (*pb.Done, error) {
	stream, err := c.cli.Generate(ctx, req)
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
		case *pb.GenerateEvent_TokenId:
			if onToken != nil {
				onToken(x.TokenId)
			}
		case *pb.GenerateEvent_Done:
			return x.Done, nil
		}
	}
}

func (c *Client) GetModel(ctx context.Context) (*pb.ModelInfo, error) {
	return c.cli.GetModel(ctx, &pb.ModelRequest{})
}
