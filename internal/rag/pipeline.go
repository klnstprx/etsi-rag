// Package rag implements the Retrieval-Augmented Generation pipeline.
package rag

import (
	"context"
	"fmt"
	"strings"
	"sync"

	"github.com/openai/openai-go"

	"etsi-rag/internal/chunking"
	"etsi-rag/internal/config"
	"etsi-rag/internal/domain"
	"etsi-rag/internal/llm"
	"etsi-rag/internal/store"
	"etsi-rag/internal/tokencount"
)

// RetrievalResult represents chunks retrieved for a query.
type RetrievalResult struct {
	Chunks    []ScoredChunk
	Documents []domain.Document
}

// ScoredChunk is a chunk with its similarity score.
type ScoredChunk struct {
	Chunk domain.Chunk
	Score float32
}

// ChatResult is the result of a RAG chat operation.
type ChatResult struct {
	Response string
	Sources  []SourceReference
}

// SourceReference identifies a source document.
type SourceReference struct {
	ID         string
	Title      string
	URL        *string
	ChunkIndex int
	Score      float32
}

// Pipeline orchestrates the RAG process.
type Pipeline struct {
	mu       sync.RWMutex
	store    store.Store
	embedder llm.LLM
	chatter  llm.LLM
	chunker  chunking.Chunker
	counter  tokencount.Counter
	config   config.RAGConfig
}

// SetEmbedder swaps the embedding client (thread-safe).
func (p *Pipeline) SetEmbedder(e llm.LLM) {
	p.mu.Lock()
	p.embedder = e
	p.mu.Unlock()
}

// SetChatter swaps the chat client (thread-safe).
func (p *Pipeline) SetChatter(c llm.LLM) {
	p.mu.Lock()
	p.chatter = c
	p.mu.Unlock()
}

// SetChunker swaps the chunker (thread-safe).
func (p *Pipeline) SetChunker(ch chunking.Chunker) {
	p.mu.Lock()
	p.chunker = ch
	p.mu.Unlock()
}

// SetCounter swaps the token counter (thread-safe).
func (p *Pipeline) SetCounter(c tokencount.Counter) {
	p.mu.Lock()
	p.counter = c
	p.mu.Unlock()
}

func (p *Pipeline) getEmbedder() llm.LLM {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.embedder
}

func (p *Pipeline) getChatter() llm.LLM {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.chatter
}

func (p *Pipeline) getChunker() chunking.Chunker {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.chunker
}

func (p *Pipeline) getCounter() tokencount.Counter {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.counter
}

// NewPipeline creates a new RAG pipeline.
// Counter may be nil at startup (misconfigured backend); operations requiring
// token counting will return errors until a valid counter is set via SetCounter.
func NewPipeline(s store.Store, embedder, chatter llm.LLM, ch chunking.Chunker, counter tokencount.Counter, cfg config.RAGConfig) *Pipeline {
	return &Pipeline{
		store:    s,
		embedder: embedder,
		chatter:  chatter,
		chunker:  ch,
		counter:  counter,
		config:   cfg,
	}
}

// Preflight checks that the embedding backend is configured and reachable.
func (p *Pipeline) Preflight(ctx context.Context) error {
	embedder := p.getEmbedder()
	if embedder == nil {
		return fmt.Errorf("embedding backend not configured")
	}
	_, err := embedder.Embed(ctx, "ping")
	if err != nil {
		return fmt.Errorf("embedding backend unreachable: %w", err)
	}
	return nil
}

// Retrieve finds relevant chunks for a query.
func (p *Pipeline) Retrieve(ctx context.Context, query string, language *string, topK *int) (*RetrievalResult, error) {
	embedder := p.getEmbedder()
	if embedder == nil {
		return nil, fmt.Errorf("embedding backend not configured")
	}

	embedding, err := embedder.Embed(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("embedding query: %w", err)
	}

	k := p.config.TopK
	if topK != nil {
		k = *topK
	}

	var lang *domain.Language
	if language != nil {
		l := domain.Language(*language)
		if l.Valid() {
			lang = &l
		}
	}

	results, err := p.store.SearchSimilar(ctx, embedding, k, lang)
	if err != nil {
		return nil, fmt.Errorf("searching: %w", err)
	}

	// Filter by threshold and build result
	var chunks []ScoredChunk
	seen := map[string]bool{}
	var docs []domain.Document

	for _, r := range results {
		if float64(r.Score) < p.config.SimilarityThreshold {
			continue
		}
		chunks = append(chunks, ScoredChunk{
			Chunk: r.Chunk,
			Score: r.Score,
		})
		if !seen[r.Document.ID] {
			seen[r.Document.ID] = true
			docs = append(docs, r.Document)
		}
	}

	return &RetrievalResult{Chunks: chunks, Documents: docs}, nil
}

// BuildContext constructs the context string from retrieval results.
func (p *Pipeline) BuildContext(ctx context.Context, results *RetrievalResult) (string, error) {
	if results == nil || len(results.Chunks) == 0 {
		return "", nil
	}

	counter := p.getCounter()
	if counter == nil {
		return "", fmt.Errorf("token counter not configured")
	}

	// Map document IDs to titles
	docTitles := make(map[string]string)
	for _, d := range results.Documents {
		docTitles[d.ID] = d.Title
	}

	var b strings.Builder
	currentTokens := 0

	for i, sc := range results.Chunks {
		title := docTitles[sc.Chunk.DocID]
		header := title
		if sc.Chunk.SectionTitle != "" {
			header = title + " > " + sc.Chunk.SectionTitle
		}
		block := fmt.Sprintf("[%d] %s:\n%s\n\n", i+1, header, sc.Chunk.Content)

		blockTokens, err := counter.Count(ctx, block)
		if err != nil {
			return "", fmt.Errorf("counting context tokens: %w", err)
		}

		if currentTokens+blockTokens > p.config.MaxContextTokens {
			break
		}
		b.WriteString(block)
		currentTokens += blockTokens
	}

	return b.String(), nil
}

// BuildMessages constructs OpenAI messages with RAG context.
func (p *Pipeline) BuildMessages(
	messages []openai.ChatCompletionMessageParamUnion,
	ragContext string,
	language string,
) []openai.ChatCompletionMessageParamUnion {
	systemPrompt := p.config.SystemPromptES
	if language == "en" {
		systemPrompt = p.config.SystemPromptEN
	}

	if ragContext != "" {
		systemPrompt += "\n\nContexto relevante:\n" + ragContext
	}

	result := make([]openai.ChatCompletionMessageParamUnion, 0, len(messages)+1)
	result = append(result, openai.ChatCompletionMessageParamUnion{
		OfSystem: &openai.ChatCompletionSystemMessageParam{
			Content: openai.ChatCompletionSystemMessageParamContentUnion{
				OfString: openai.String(systemPrompt),
			},
		},
	})
	result = append(result, messages...)

	return result
}

// Chat executes the full RAG pipeline for a chat request.
func (p *Pipeline) Chat(
	ctx context.Context,
	messages []openai.ChatCompletionMessageParamUnion,
	language string,
	topK *int,
) (*ChatResult, error) {
	// Extract query from last user message
	query := extractLastUserMessage(messages)
	if query == "" {
		return nil, fmt.Errorf("no user message found")
	}

	// Retrieve relevant chunks
	retrieval, err := p.Retrieve(ctx, query, &language, topK)
	if err != nil {
		return nil, fmt.Errorf("retrieval: %w", err)
	}

	// Build context and augment messages
	ragContext, err := p.BuildContext(ctx, retrieval)
	if err != nil {
		return nil, fmt.Errorf("building context: %w", err)
	}
	augmented := p.BuildMessages(messages, ragContext, language)

	chatter := p.getChatter()
	if chatter == nil {
		return nil, fmt.Errorf("chat backend not configured")
	}

	// Generate response
	response, err := chatter.Chat(ctx, augmented)
	if err != nil {
		return nil, fmt.Errorf("generating response: %w", err)
	}

	// Build source references
	var sources []SourceReference
	docMap := make(map[string]domain.Document)
	for _, d := range retrieval.Documents {
		docMap[d.ID] = d
	}
	for _, sc := range retrieval.Chunks {
		doc := docMap[sc.Chunk.DocID]
		sources = append(sources, SourceReference{
			ID:         doc.ID,
			Title:      doc.Title,
			URL:        doc.URL,
			ChunkIndex: sc.Chunk.ChunkIndex,
			Score:      sc.Score,
		})
	}

	return &ChatResult{Response: response, Sources: sources}, nil
}

// Ingest adds documents to the RAG index.
// Returns (documents ingested, chunks created, error).
func (p *Pipeline) Ingest(ctx context.Context, docs []domain.Document) (int, int, error) {
	embedder := p.getEmbedder()
	if embedder == nil {
		return 0, 0, fmt.Errorf("embedding backend not configured")
	}

	var totalDocs, totalChunks int

	for _, doc := range docs {
		if err := p.store.CreateDocument(ctx, &doc); err != nil {
			return totalDocs, totalChunks, fmt.Errorf("storing document %s: %w", doc.ID, err)
		}

		chunks, err := p.getChunker().Chunk(ctx, doc.Content)
		if err != nil {
			return totalDocs, totalChunks, fmt.Errorf("chunking document %s: %w", doc.ID, err)
		}
		if len(chunks) == 0 {
			totalDocs++
			continue
		}

		// Convert chunking.Chunk to domain.Chunk and collect texts
		domainChunks := make([]domain.Chunk, len(chunks))
		texts := make([]string, len(chunks))
		for i, c := range chunks {
			domainChunks[i] = domain.Chunk{
				Content:      c.Content,
				ChunkIndex:   c.Index,
				SectionTitle: c.SectionTitle,
			}
			texts[i] = c.Content
		}

		embeddings, err := embedder.EmbedBatch(ctx, texts)
		if err != nil {
			return totalDocs, totalChunks, fmt.Errorf("embedding chunks for %s: %w", doc.ID, err)
		}

		if err := p.store.CreateChunks(ctx, doc.ID, domainChunks, embeddings); err != nil {
			return totalDocs, totalChunks, fmt.Errorf("storing chunks for %s: %w", doc.ID, err)
		}

		totalDocs++
		totalChunks += len(chunks)
	}

	return totalDocs, totalChunks, nil
}

// extractLastUserMessage gets the content of the last user message.
func extractLastUserMessage(messages []openai.ChatCompletionMessageParamUnion) string {
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].OfUser != nil {
			if messages[i].OfUser.Content.OfString.Valid() {
				return messages[i].OfUser.Content.OfString.Value
			}
		}
	}
	return ""
}
