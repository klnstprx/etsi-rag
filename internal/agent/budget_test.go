package agent

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/openai/openai-go"

	"etsi-rag/internal/tokencount"
)

func TestBudget_ToolPool(t *testing.T) {
	tests := []struct {
		name string
		b    Budget
		want int
	}{
		{
			name: "typical: window minus reserves",
			b:    Budget{ContextWindow: 8192, Generation: 1024, Fixed: 600, UserReserve: 512, HistoryReserve: 1280},
			want: 8192 - 1024 - 600 - 512 - 1280,
		},
		{
			name: "no reserves",
			b:    Budget{ContextWindow: 4096, Generation: 0, Fixed: 0, UserReserve: 0, HistoryReserve: 0},
			want: 4096,
		},
		{
			name: "over-subscribed clamps to zero, never negative",
			b:    Budget{ContextWindow: 1000, Generation: 800, Fixed: 400, UserReserve: 100, HistoryReserve: 100},
			want: 0,
		},
		{
			name: "zero window clamps to zero",
			b:    Budget{ContextWindow: 0, Generation: 1024, Fixed: 100},
			want: 0,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.b.ToolPool(); got != tt.want {
				t.Errorf("ToolPool() = %d, want %d", got, tt.want)
			}
		})
	}
}

// quarterCounter counts len(text)/4 tokens — a deterministic stand-in for the
// real BPE counters in budget measurement tests.
type quarterCounter struct{}

func (quarterCounter) Count(_ context.Context, text string) (int, error) {
	return len(text) / 4, nil
}
func (quarterCounter) Tokenize(_ context.Context, text string) ([]int, error) { return nil, nil }
func (quarterCounter) Detokenize(_ context.Context, _ []int) (string, error)  { return "", nil }

// errCounter always errors — used to assert MeasureFixed propagates the error
// to the caller rather than swallowing it.
type errCounter struct{}

func (errCounter) Count(_ context.Context, _ string) (int, error) {
	return 0, errors.New("counter unavailable")
}
func (errCounter) Tokenize(_ context.Context, _ string) ([]int, error) { return nil, nil }
func (errCounter) Detokenize(_ context.Context, _ []int) (string, error) {
	return "", nil
}

func TestMeasureFixed(t *testing.T) {
	system := strings.Repeat("a", 400) // 100 tokens at len/4
	tool := newStubTool("search", "Search the corpus", func(_ context.Context, _ string) (ToolResult, error) {
		return ToolResult{}, nil
	})

	tests := []struct {
		name    string
		counter tokencount.Counter
		system  string
		tools   []Tool
		wantErr bool
		// wantMin asserts the result is at least this (margin applied, rounded up).
		wantMin int
	}{
		{
			name:    "system only, no tools",
			counter: quarterCounter{},
			system:  system,
			tools:   nil,
			// 100 tokens * 1.10 = 110
			wantMin: 110,
		},
		{
			name:    "system plus a tool schema",
			counter: quarterCounter{},
			system:  system,
			tools:   []Tool{tool},
			// system (100) + schema tokens, both * 1.10 → strictly greater than system-only
			wantMin: 111,
		},
		{
			name:    "counter error propagates",
			counter: errCounter{},
			system:  system,
			tools:   []Tool{tool},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := MeasureFixed(context.Background(), tt.counter, tt.system, tt.tools)
			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got < tt.wantMin {
				t.Errorf("MeasureFixed() = %d, want >= %d", got, tt.wantMin)
			}
		})
	}
}

// TestMeasureFixed_CountsToolSchema asserts the tool definition JSON
// contributes to the measured cost (a tool with a long description costs more
// than a bare one).
func TestMeasureFixed_CountsToolSchema(t *testing.T) {
	short := newStubTool("s", "x", func(_ context.Context, _ string) (ToolResult, error) {
		return ToolResult{}, nil
	})
	long := &stubTool{
		name:        "long_tool_name",
		description: strings.Repeat("describe this tool thoroughly ", 20),
		parameters:  openai.FunctionParameters{"type": "object", "properties": map[string]any{}},
		execute: func(_ context.Context, _ string) (ToolResult, error) {
			return ToolResult{}, nil
		},
	}

	c := quarterCounter{}
	withShort, err := MeasureFixed(context.Background(), c, "", []Tool{short})
	if err != nil {
		t.Fatalf("MeasureFixed short: %v", err)
	}
	withLong, err := MeasureFixed(context.Background(), c, "", []Tool{long})
	if err != nil {
		t.Fatalf("MeasureFixed long: %v", err)
	}
	if withLong <= withShort {
		t.Errorf("expected long tool schema to cost more: long=%d short=%d", withLong, withShort)
	}
}
