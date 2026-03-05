package rag

import (
	"context"
	"testing"

	"github.com/openai/openai-go"

	"etsi-rag/internal/chunking"
	"etsi-rag/internal/config"
	"etsi-rag/internal/domain"
	"etsi-rag/internal/store"
)

// mockLLM implements llm.LLM for testing.
type mockLLM struct {
	embedFn      func(ctx context.Context, text string) ([]float32, error)
	embedBatchFn func(ctx context.Context, texts []string) ([][]float32, error)
	chatFn       func(ctx context.Context, messages []openai.ChatCompletionMessageParamUnion) (string, error)
}

func (m *mockLLM) Embed(ctx context.Context, text string) ([]float32, error) {
	return m.embedFn(ctx, text)
}

func (m *mockLLM) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	return m.embedBatchFn(ctx, texts)
}

func (m *mockLLM) Chat(ctx context.Context, messages []openai.ChatCompletionMessageParamUnion) (string, error) {
	return m.chatFn(ctx, messages)
}

func (m *mockLLM) ChatStream(ctx context.Context, messages []openai.ChatCompletionMessageParamUnion) (<-chan string, <-chan error) {
	tokens := make(chan string)
	errc := make(chan error)
	close(tokens)
	close(errc)
	return tokens, errc
}

func setupTestPipeline(t *testing.T) (*Pipeline, *store.SQLiteStore) {
	t.Helper()

	db, err := store.OpenDB(":memory:")
	if err != nil {
		t.Fatalf("opening test db: %v", err)
	}
	if err := store.Migrate(db, 4); err != nil {
		db.Close()
		t.Fatalf("migrating: %v", err)
	}

	s := store.NewSQLiteStore(db, 4)
	chunker := chunking.NewRecursiveCharacterChunker(config.ChunkingConfig{
		ChunkSize:    100,
		ChunkOverlap: 0,
	})

	mock := &mockLLM{
		embedFn: func(ctx context.Context, text string) ([]float32, error) {
			return []float32{0.5, 0.5, 0.0, 0.0}, nil
		},
		embedBatchFn: func(ctx context.Context, texts []string) ([][]float32, error) {
			result := make([][]float32, len(texts))
			for i := range texts {
				result[i] = []float32{0.5, 0.5, 0.0, 0.0}
			}
			return result, nil
		},
		chatFn: func(ctx context.Context, messages []openai.ChatCompletionMessageParamUnion) (string, error) {
			return "Test response from LLM", nil
		},
	}

	p := NewPipeline(s, mock, mock, chunker, config.RAGConfig{
		TopK:                5,
		SimilarityThreshold: 0.0, // accept all
		MaxContextTokens:    1024,
		SystemPromptES:      "Eres un asistente.",
		SystemPromptEN:      "You are an assistant.",
	})

	return p, s
}

func TestIngest(t *testing.T) {
	p, _ := setupTestPipeline(t)
	ctx := context.Background()

	docs := []domain.Document{
		{
			ID:       "doc-1",
			Title:    "Test Doc",
			Content:  "This is test content for ingestion.",
			Language: domain.LanguageES,
			Category: domain.CategoryGeneral,
		},
	}

	ingested, chunks, err := p.Ingest(ctx, docs)
	if err != nil {
		t.Fatalf("Ingest failed: %v", err)
	}
	if ingested != 1 {
		t.Errorf("expected 1 ingested, got %d", ingested)
	}
	if chunks != 1 {
		t.Errorf("expected 1 chunk, got %d", chunks)
	}
}

func TestRetrieve(t *testing.T) {
	p, s := setupTestPipeline(t)
	ctx := context.Background()

	// Manually insert a document with known embedding
	doc := &domain.Document{
		ID:       "doc-1",
		Title:    "Test Doc",
		Content:  "Relevant content",
		Language: domain.LanguageES,
		Category: domain.CategoryGeneral,
	}
	if err := s.CreateDocument(ctx, doc); err != nil {
		t.Fatal(err)
	}
	chunks := []domain.Chunk{{Content: "Relevant content", ChunkIndex: 0}}
	embeddings := [][]float32{{0.5, 0.5, 0.0, 0.0}}
	if err := s.CreateChunks(ctx, "doc-1", chunks, embeddings); err != nil {
		t.Fatal(err)
	}

	lang := "es"
	result, err := p.Retrieve(ctx, "query", &lang, nil)
	if err != nil {
		t.Fatalf("Retrieve failed: %v", err)
	}

	if len(result.Chunks) == 0 {
		t.Fatal("expected chunks in result")
	}
	if len(result.Documents) == 0 {
		t.Fatal("expected documents in result")
	}
}

func TestBuildContext(t *testing.T) {
	p, _ := setupTestPipeline(t)

	result := &RetrievalResult{
		Chunks: []ScoredChunk{
			{Chunk: domain.Chunk{DocID: "doc-1", Content: "chunk text", ChunkIndex: 0}, Score: 0.9},
		},
		Documents: []domain.Document{
			{ID: "doc-1", Title: "Test Document"},
		},
	}

	ctx := p.BuildContext(result)
	if ctx == "" {
		t.Fatal("expected non-empty context")
	}
	if !contains(ctx, "Test Document") {
		t.Error("context should contain document title")
	}
	if !contains(ctx, "chunk text") {
		t.Error("context should contain chunk content")
	}
}

func TestBuildMessages(t *testing.T) {
	p, _ := setupTestPipeline(t)

	messages := []openai.ChatCompletionMessageParamUnion{
		{
			OfUser: &openai.ChatCompletionUserMessageParam{
				Content: openai.ChatCompletionUserMessageParamContentUnion{
					OfString: openai.String("What programs are available?"),
				},
			},
		},
	}

	result := p.BuildMessages(messages, "context text", "es")

	// Should have system message + original message
	if len(result) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(result))
	}

	// First should be system
	if result[0].OfSystem == nil {
		t.Fatal("first message should be system")
	}
}

func TestChat(t *testing.T) {
	p, s := setupTestPipeline(t)
	ctx := context.Background()

	// Insert test data
	doc := &domain.Document{
		ID:       "doc-1",
		Title:    "Test Doc",
		Content:  "Relevant info",
		Language: domain.LanguageES,
		Category: domain.CategoryGeneral,
	}
	if err := s.CreateDocument(ctx, doc); err != nil {
		t.Fatal(err)
	}
	if err := s.CreateChunks(ctx, "doc-1",
		[]domain.Chunk{{Content: "Relevant info", ChunkIndex: 0}},
		[][]float32{{0.5, 0.5, 0.0, 0.0}},
	); err != nil {
		t.Fatal(err)
	}

	messages := []openai.ChatCompletionMessageParamUnion{
		{
			OfUser: &openai.ChatCompletionUserMessageParam{
				Content: openai.ChatCompletionUserMessageParamContentUnion{
					OfString: openai.String("Tell me about this"),
				},
			},
		},
	}

	result, err := p.Chat(ctx, messages, "es", nil)
	if err != nil {
		t.Fatalf("Chat failed: %v", err)
	}

	if result.Response == "" {
		t.Error("expected non-empty response")
	}
	if len(result.Sources) == 0 {
		t.Error("expected sources")
	}
}

func TestExtractLastUserMessage(t *testing.T) {
	messages := []openai.ChatCompletionMessageParamUnion{
		{
			OfSystem: &openai.ChatCompletionSystemMessageParam{
				Content: openai.ChatCompletionSystemMessageParamContentUnion{
					OfString: openai.String("System prompt"),
				},
			},
		},
		{
			OfUser: &openai.ChatCompletionUserMessageParam{
				Content: openai.ChatCompletionUserMessageParamContentUnion{
					OfString: openai.String("First question"),
				},
			},
		},
		{
			OfUser: &openai.ChatCompletionUserMessageParam{
				Content: openai.ChatCompletionUserMessageParamContentUnion{
					OfString: openai.String("Second question"),
				},
			},
		},
	}

	got := extractLastUserMessage(messages)
	if got != "Second question" {
		t.Errorf("expected 'Second question', got %q", got)
	}
}

func TestExtractLastUserMessageEmpty(t *testing.T) {
	messages := []openai.ChatCompletionMessageParamUnion{
		{
			OfSystem: &openai.ChatCompletionSystemMessageParam{
				Content: openai.ChatCompletionSystemMessageParamContentUnion{
					OfString: openai.String("System only"),
				},
			},
		},
	}

	got := extractLastUserMessage(messages)
	if got != "" {
		t.Errorf("expected empty string, got %q", got)
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > 0 && containsStr(s, substr))
}

func containsStr(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
