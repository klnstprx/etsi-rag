package store

import (
	"context"
	"errors"
	"testing"

	"etsi-rag/internal/domain"
)

func TestArtifactCRUD(t *testing.T) {
	store := setupTestStore(t)
	ctx := context.Background()

	doc := &domain.Document{
		ID:       "doc-1",
		Title:    "Test Document",
		Content:  "x",
		Language: domain.LanguageES,
		Category: domain.CategoryGeneral,
	}
	if err := store.CreateDocument(ctx, doc); err != nil {
		t.Fatalf("creating document: %v", err)
	}

	t.Run("upsert and get by ref", func(t *testing.T) {
		a := &domain.Artifact{
			DocID:          "doc-1",
			Ref:            "p1",
			Kind:           domain.ArtifactPDFPage,
			ObjectStoreKey: "crawl/img/abc/p1.png",
			MimeType:       "image/png",
			Width:          800,
			Height:         1100,
		}
		if err := store.UpsertArtifact(ctx, a); err != nil {
			t.Fatalf("upserting: %v", err)
		}
		if a.ID == 0 {
			t.Fatal("expected ID to be assigned")
		}
		if a.PublicToken == "" {
			t.Fatal("expected public token to be assigned")
		}

		got, err := store.GetArtifact(ctx, "doc-1", "p1")
		if err != nil {
			t.Fatalf("get by ref: %v", err)
		}
		if got.ID != a.ID || got.PublicToken != a.PublicToken {
			t.Errorf("mismatch on round-trip: got %+v want %+v", got, a)
		}
		if got.Width != 800 || got.Height != 1100 {
			t.Errorf("dims mismatch: got %dx%d", got.Width, got.Height)
		}
	})

	t.Run("get by public token", func(t *testing.T) {
		a := &domain.Artifact{
			DocID:          "doc-1",
			Ref:            "full",
			Kind:           domain.ArtifactHTMLScreenshot,
			ObjectStoreKey: "crawl/img/abc/full.png",
			MimeType:       "image/png",
		}
		if err := store.UpsertArtifact(ctx, a); err != nil {
			t.Fatalf("upserting: %v", err)
		}
		got, err := store.GetArtifactByToken(ctx, a.PublicToken)
		if err != nil {
			t.Fatalf("get by token: %v", err)
		}
		if got.Kind != domain.ArtifactHTMLScreenshot {
			t.Errorf("kind = %q, want html_screenshot", got.Kind)
		}
	})

	t.Run("upsert is idempotent on (doc_id, ref)", func(t *testing.T) {
		first := &domain.Artifact{
			DocID:          "doc-1",
			Ref:            "p1",
			Kind:           domain.ArtifactPDFPage,
			ObjectStoreKey: "crawl/img/abc/p1.png",
			MimeType:       "image/png",
		}
		// already inserted earlier; second upsert with same (doc_id, ref) should overwrite key
		first.ObjectStoreKey = "crawl/img/abc/p1-v2.png"
		if err := store.UpsertArtifact(ctx, first); err != nil {
			t.Fatalf("re-upsert: %v", err)
		}
		got, err := store.GetArtifact(ctx, "doc-1", "p1")
		if err != nil {
			t.Fatalf("get after re-upsert: %v", err)
		}
		if got.ObjectStoreKey != "crawl/img/abc/p1-v2.png" {
			t.Errorf("expected updated key, got %q", got.ObjectStoreKey)
		}
	})

	t.Run("missing returns ErrNotFound", func(t *testing.T) {
		_, err := store.GetArtifact(ctx, "doc-1", "nope")
		if !errors.Is(err, ErrNotFound) {
			t.Errorf("expected ErrNotFound, got %v", err)
		}
		_, err = store.GetArtifactByToken(ctx, "no-such-token")
		if !errors.Is(err, ErrNotFound) {
			t.Errorf("expected ErrNotFound for token, got %v", err)
		}
	})
}

func TestArtifactCascadesOnDocDelete(t *testing.T) {
	store := setupTestStore(t)
	ctx := context.Background()

	doc := &domain.Document{
		ID:       "doc-1",
		Title:    "x",
		Content:  "x",
		Language: domain.LanguageES,
		Category: domain.CategoryGeneral,
	}
	if err := store.CreateDocument(ctx, doc); err != nil {
		t.Fatalf("creating document: %v", err)
	}

	a := &domain.Artifact{
		DocID:          "doc-1",
		Ref:            "p1",
		Kind:           domain.ArtifactPDFPage,
		ObjectStoreKey: "crawl/img/abc/p1.png",
		MimeType:       "image/png",
	}
	if err := store.UpsertArtifact(ctx, a); err != nil {
		t.Fatalf("upserting: %v", err)
	}

	if err := store.DeleteDocument(ctx, "doc-1"); err != nil {
		t.Fatalf("deleting document: %v", err)
	}

	if _, err := store.GetArtifact(ctx, "doc-1", "p1"); !errors.Is(err, ErrNotFound) {
		t.Errorf("expected ErrNotFound after cascade, got %v", err)
	}
}

func TestPublicTokensAreUnique(t *testing.T) {
	store := setupTestStore(t)
	ctx := context.Background()

	doc := &domain.Document{
		ID:       "doc-1",
		Title:    "x",
		Content:  "x",
		Language: domain.LanguageES,
		Category: domain.CategoryGeneral,
	}
	if err := store.CreateDocument(ctx, doc); err != nil {
		t.Fatalf("creating document: %v", err)
	}

	tokens := make(map[string]struct{})
	for i := range 32 {
		a := &domain.Artifact{
			DocID:          "doc-1",
			Ref:            "p" + itoa(i),
			Kind:           domain.ArtifactPDFPage,
			ObjectStoreKey: "k",
			MimeType:       "image/png",
		}
		if err := store.UpsertArtifact(ctx, a); err != nil {
			t.Fatalf("upserting: %v", err)
		}
		if _, dup := tokens[a.PublicToken]; dup {
			t.Fatalf("duplicate token: %s", a.PublicToken)
		}
		tokens[a.PublicToken] = struct{}{}
	}
}

func itoa(i int) string {
	const digits = "0123456789"
	if i == 0 {
		return "0"
	}
	var b [20]byte
	pos := len(b)
	for i > 0 {
		pos--
		b[pos] = digits[i%10]
		i /= 10
	}
	return string(b[pos:])
}
