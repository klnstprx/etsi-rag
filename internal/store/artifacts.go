package store

import (
	"context"
	"crypto/rand"
	"database/sql"
	"encoding/hex"
	"errors"
	"fmt"

	"etsi-rag/internal/domain"
)

// generatePublicToken returns a 128-bit random hex token (32 chars), used as
// the opaque path component in /v1/artifacts/{token}.png to avoid trivial
// enumeration of artifact IDs.
func generatePublicToken() (string, error) {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", fmt.Errorf("generating public token: %w", err)
	}
	return hex.EncodeToString(b[:]), nil
}

// UpsertArtifact inserts a new artifact row or updates an existing one keyed
// by (doc_id, ref). On insert, ID and PublicToken are populated on the
// passed-in struct.
func (s *SQLiteStore) UpsertArtifact(ctx context.Context, a *domain.Artifact) error {
	existing, err := s.Artifact(ctx, a.DocID, a.Ref)
	if err != nil && !errors.Is(err, ErrNotFound) {
		return fmt.Errorf("checking existing artifact: %w", err)
	}

	if existing != nil {
		// Update in place; preserve ID and public token.
		_, err := s.db.ExecContext(ctx, `
			UPDATE document_artifacts
			SET kind = ?, object_store_key = ?, mime_type = ?, width = ?, height = ?
			WHERE id = ?
		`, string(a.Kind), a.ObjectStoreKey, a.MimeType, a.Width, a.Height, existing.ID)
		if err != nil {
			return fmt.Errorf("updating artifact: %w", err)
		}
		a.ID = existing.ID
		a.PublicToken = existing.PublicToken
		return nil
	}

	token, err := generatePublicToken()
	if err != nil {
		return err
	}

	res, err := s.db.ExecContext(ctx, `
		INSERT INTO document_artifacts
			(doc_id, ref, kind, object_store_key, mime_type, width, height, public_token)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?)
	`, a.DocID, a.Ref, string(a.Kind), a.ObjectStoreKey, a.MimeType, a.Width, a.Height, token)
	if err != nil {
		return fmt.Errorf("inserting artifact: %w", err)
	}
	id, err := res.LastInsertId()
	if err != nil {
		return fmt.Errorf("getting artifact id: %w", err)
	}
	a.ID = id
	a.PublicToken = token
	return nil
}

const artifactSelect = `
	SELECT id, doc_id, ref, kind, object_store_key, mime_type, width, height, public_token
	FROM document_artifacts
`

// Artifact looks up an artifact by (doc_id, ref). Returns ErrNotFound when
// no row matches.
func (s *SQLiteStore) Artifact(ctx context.Context, docID, ref string) (*domain.Artifact, error) {
	return s.queryArtifact(ctx, " WHERE doc_id = ? AND ref = ?", docID, ref)
}

// ArtifactByToken looks up an artifact by its opaque public token (used by
// the /v1/artifacts/{token}.png route).
func (s *SQLiteStore) ArtifactByToken(ctx context.Context, token string) (*domain.Artifact, error) {
	return s.queryArtifact(ctx, " WHERE public_token = ?", token)
}

// queryArtifact returns the artifact matching where, or ErrNotFound.
func (s *SQLiteStore) queryArtifact(ctx context.Context, where string, args ...any) (*domain.Artifact, error) {
	var a domain.Artifact
	err := s.db.QueryRowContext(ctx, artifactSelect+where, args...).
		Scan(&a.ID, &a.DocID, &a.Ref, &a.Kind, &a.ObjectStoreKey, &a.MimeType, &a.Width, &a.Height, &a.PublicToken)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, ErrNotFound
	}
	if err != nil {
		return nil, fmt.Errorf("scanning artifact: %w", err)
	}
	return &a, nil
}

// CountArtifacts returns the total number of stored artifact rows. Useful for
// the admin dashboard counter.
func (s *SQLiteStore) CountArtifacts(ctx context.Context) (int, error) {
	return s.count(ctx, "document_artifacts")
}
