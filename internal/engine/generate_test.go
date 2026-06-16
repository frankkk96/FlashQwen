package engine

import (
	"errors"
	"testing"
)

// resolveMaxTokens only reads g.maxCtx, so a bare Generator is enough to test the length policy.
func TestResolveMaxTokens(t *testing.T) {
	c := &Generator{maxCtx: 100}
	cases := []struct {
		name      string
		promptLen int
		requested int
		want      int
		wantErr   bool
	}{
		{"omitted fills remaining window", 10, 0, 90, false},
		{"explicit under room is kept", 10, 50, 50, false},
		{"explicit over room is clamped", 10, 1000, 90, false},
		{"explicit exactly room is kept", 10, 90, 90, false},
		{"negative treated as omitted", 30, -5, 70, false},
		{"prompt fills the window", 100, 16, 0, true},
		{"prompt over the window", 120, 16, 0, true},
		{"one token of room", 99, 0, 1, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := c.resolveMaxTokens(tc.promptLen, tc.requested)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error, got %d", got)
				}
				if !errors.Is(err, ErrPromptTooLong) {
					t.Errorf("error %v is not ErrPromptTooLong", err)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tc.want {
				t.Errorf("resolveMaxTokens(%d, %d) = %d, want %d", tc.promptLen, tc.requested, got, tc.want)
			}
		})
	}
}
