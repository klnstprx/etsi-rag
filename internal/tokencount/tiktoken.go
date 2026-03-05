package tokencount

import (
	"context"
	"fmt"

	"github.com/pkoukk/tiktoken-go"
)

// TiktokenCounter uses tiktoken for OpenAI-compatible token counting.
type TiktokenCounter struct {
	enc *tiktoken.Tiktoken
}

// NewTiktokenCounter creates a counter using tiktoken encoding for the given model.
func NewTiktokenCounter(model string) (*TiktokenCounter, error) {
	enc, err := tiktoken.EncodingForModel(model)
	if err != nil {
		return nil, fmt.Errorf("tiktoken encoding for model %q: %w", model, err)
	}
	return &TiktokenCounter{enc: enc}, nil
}

func (t *TiktokenCounter) Count(_ context.Context, text string) (int, error) {
	return len(t.enc.Encode(text, nil, nil)), nil
}

func (t *TiktokenCounter) Tokenize(_ context.Context, text string) ([]int, error) {
	return t.enc.Encode(text, nil, nil), nil
}

func (t *TiktokenCounter) Detokenize(_ context.Context, tokens []int) (string, error) {
	return t.enc.Decode(tokens), nil
}
