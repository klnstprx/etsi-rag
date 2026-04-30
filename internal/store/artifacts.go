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
	existing, err := s.GetArtifact(ctx, a.DocID, a.Ref)
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

func scanArtifact(row interface {
	Scan(dest ...any) error
}) (*domain.Artifact, error) {
	var a domain.Artifact
	var kind string
	if err := row.Scan(&a.ID, &a.DocID, &a.Ref, &kind, &a.ObjectStoreKey, &a.MimeType, &a.Width, &a.Height, &a.PublicToken); err != nil {
		return nil, err
	}
	a.Kind = domain.ArtifactKind(kind)
	return &a, nil
}

// GetArtifact looks up an artifact by (doc_id, ref). Returns ErrNotFound when
// no row matches.
func (s *SQLiteStore) GetArtifact(ctx context.Context, docID, ref string) (*domain.Artifact, error) {
	row := s.db.QueryRowContext(ctx, artifactSelect+" WHERE doc_id = ? AND ref = ?", docID, ref)
	a, err := scanArtifact(row)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, ErrNotFound
	}
	if err != nil {
		return nil, fmt.Errorf("scanning artifact: %w", err)
	}
	return a, nil
}

// GetArtifactByID looks up an artifact by primary key.
func (s *SQLiteStore) GetArtifactByID(ctx context.Context, id int64) (*domain.Artifact, error) {
	row := s.db.QueryRowContext(ctx, artifactSelect+" WHERE id = ?", id)
	a, err := scanArtifact(row)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, ErrNotFound
	}
	if err != nil {
		return nil, fmt.Errorf("scanning artifact: %w", err)
	}
	return a, nil
}

// GetArtifactByToken looks up an artifact by its opaque public token (used by
// the /v1/artifacts/{token}.png route).
func (s *SQLiteStore) GetArtifactByToken(ctx context.Context, token string) (*domain.Artifact, error) {
	row := s.db.QueryRowContext(ctx, artifactSelect+" WHERE public_token = ?", token)
	a, err := scanArtifact(row)
	if errors.Is(err, sql.ErrNoRows) {
		return nil, ErrNotFound
	}
	if err != nil {
		return nil, fmt.Errorf("scanning artifact: %w", err)
	}
	return a, nil
}

// ListArtifacts returns all artifacts for a given document, ordered by ref.
func (s *SQLiteStore) ListArtifacts(ctx context.Context, docID string) ([]domain.Artifact, error) {
	rows, err := s.db.QueryContext(ctx, artifactSelect+" WHERE doc_id = ? ORDER BY ref", docID)
	if err != nil {
		return nil, fmt.Errorf("querying artifacts: %w", err)
	}
	defer rows.Close()

	var out []domain.Artifact
	for rows.Next() {
		a, err := scanArtifact(rows)
		if err != nil {
			return nil, fmt.Errorf("scanning artifact: %w", err)
		}
		out = append(out, *a)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("iterating artifacts: %w", err)
	}
	return out, nil
}

// CountArtifacts returns the total number of stored artifact rows. Useful for
// the admin dashboard counter.
func (s *SQLiteStore) CountArtifacts(ctx context.Context) (int, error) {
	var n int
	if err := s.db.QueryRowContext(ctx, "SELECT COUNT(*) FROM document_artifacts").Scan(&n); err != nil {
		return 0, fmt.Errorf("counting artifacts: %w", err)
	}
	return n, nil
}
