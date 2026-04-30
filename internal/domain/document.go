package domain

import (
	"database/sql/driver"
	"fmt"
)

// Language represents supported languages.
type Language string

const (
	LanguageES Language = "es"
	LanguageEN Language = "en"
)

// Valid returns true if the language is supported.
func (l Language) Valid() bool {
	return l == LanguageES || l == LanguageEN
}

// Value implements driver.Valuer for database storage.
func (l Language) Value() (driver.Value, error) {
	return string(l), nil
}

// Scan implements sql.Scanner for database retrieval.
func (l *Language) Scan(value interface{}) error {
	if value == nil {
		*l = LanguageES
		return nil
	}
	switch v := value.(type) {
	case string:
		*l = Language(v)
	case []byte:
		*l = Language(v)
	default:
		return fmt.Errorf("cannot scan %T into Language", value)
	}
	return nil
}

// Category represents document categories.
type Category string

const (
	CategoryDegree    Category = "degree"
	CategoryCourse    Category = "course"
	CategoryProcedure Category = "procedure"
	CategoryStaff     Category = "staff"
	CategoryNews      Category = "news"
	CategoryGeneral   Category = "general"
)

// Valid returns true if the category is supported.
func (c Category) Valid() bool {
	switch c {
	case CategoryDegree, CategoryCourse, CategoryProcedure,
		CategoryStaff, CategoryNews, CategoryGeneral:
		return true
	}
	return false
}

// Value implements driver.Valuer for database storage.
func (c Category) Value() (driver.Value, error) {
	return string(c), nil
}

// Scan implements sql.Scanner for database retrieval.
func (c *Category) Scan(value interface{}) error {
	if value == nil {
		*c = CategoryGeneral
		return nil
	}
	switch v := value.(type) {
	case string:
		*c = Category(v)
	case []byte:
		*c = Category(v)
	default:
		return fmt.Errorf("cannot scan %T into Category", value)
	}
	return nil
}

// Document represents a stored document.
type Document struct {
	ID       string
	Title    string
	Content  string
	URL      *string
	Language Language
	Category Category
}

// Chunk represents a document chunk for vector search.
type Chunk struct {
	ID           int64
	DocID        string
	Content      string
	ChunkIndex   int
	SectionTitle string
}

// SourceReference identifies a source document used in RAG context.
type SourceReference struct {
	ID         string  `json:"id"`
	Title      string  `json:"title"`
	URL        *string `json:"url,omitempty"`
	ChunkIndex int     `json:"chunk_index"`
	Score      float32 `json:"score"`
}

// ArtifactKind classifies an image artifact derived from a document.
type ArtifactKind string

const (
	ArtifactPDFPage        ArtifactKind = "pdf_page"
	ArtifactHTMLScreenshot ArtifactKind = "html_screenshot"
)

// Artifact is an image rendered from a source document (PDF page or HTML page
// screenshot) that the agent can attach to a vision-capable LLM as visual
// context. The Ref is the chunk-marker reference (e.g. "p1", "p2", "full").
type Artifact struct {
	ID             int64
	DocID          string
	Ref            string
	Kind           ArtifactKind
	ObjectStoreKey string
	MimeType       string
	Width          int
	Height         int
	PublicToken    string // opaque token used in /v1/artifacts/{token}.png
}
