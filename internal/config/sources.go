package config

import (
	"fmt"
	"net/url"
	"path/filepath"
	"strings"

	"github.com/knadh/koanf/parsers/toml"
	"github.com/knadh/koanf/providers/file"
	"github.com/knadh/koanf/providers/structs"
	"github.com/knadh/koanf/v2"

	"etsi-rag/internal/domain"
)

const (
	// SourcesFileName is the user-editable sources file (gitignored).
	SourcesFileName = "sources.toml"
	// DefaultSourcesFileName is the tracked template with default seed URLs.
	DefaultSourcesFileName = "sources.default.toml"
)

// sourcesFile is the on-disk shape of sources.toml. The wrapper exists so
// koanf can marshal a top-level [[sources]] array without a parent key clash.
type sourcesFile struct {
	Sources []domain.CrawlSource `koanf:"sources"`
}

// SourcesPath returns the full path to the user sources file within configDir.
func SourcesPath(configDir string) string {
	return filepath.Join(configDir, SourcesFileName)
}

// EnsureSourcesFile copies the default sources file to the user path if the
// user file does not exist.
func EnsureSourcesFile(configDir string) error {
	return copyIfMissing(SourcesPath(configDir), filepath.Join(configDir, DefaultSourcesFileName))
}

// LoadSourcesFile reads and validates the sources file at path. The file is
// the single source of truth for crawl_sources; crawl_pages and crawl_runs
// reference each entry by its stable Key.
func LoadSourcesFile(path string) ([]domain.CrawlSource, error) {
	k := koanf.New(".")
	if err := k.Load(file.Provider(path), toml.Parser()); err != nil {
		return nil, fmt.Errorf("loading %s: %w", path, err)
	}

	var f sourcesFile
	if err := k.Unmarshal("", &f); err != nil {
		return nil, fmt.Errorf("unmarshalling sources: %w", err)
	}

	if err := validateSources(f.Sources); err != nil {
		return nil, err
	}
	return f.Sources, nil
}

// WriteSources marshals sources to TOML and writes them atomically to path.
// Validates first so we never persist a malformed file.
func WriteSources(sources []domain.CrawlSource, path string) error {
	if err := validateSources(sources); err != nil {
		return err
	}

	k := koanf.New(".")
	if err := k.Load(structs.Provider(&sourcesFile{Sources: sources}, "koanf"), nil); err != nil {
		return fmt.Errorf("loading sources into koanf: %w", err)
	}

	data, err := k.Marshal(toml.Parser())
	if err != nil {
		return fmt.Errorf("marshaling sources: %w", err)
	}

	return writeAtomic(path, data)
}

// validateSources enforces invariants the rest of the system relies on:
// non-empty Key, non-empty parsable SeedURL, and Key uniqueness. Names need
// not be unique, but keys must be — they are FKs in crawl_pages/crawl_runs.
// Tag entries (when present) must be non-empty after trimming; the loader
// trims them in place so downstream code never sees whitespace-only tags.
func validateSources(sources []domain.CrawlSource) error {
	keys := make(map[string]struct{}, len(sources))
	for i, src := range sources {
		if strings.TrimSpace(src.Key) == "" {
			return fmt.Errorf("source[%d]: key is required", i)
		}
		if _, dup := keys[src.Key]; dup {
			return fmt.Errorf("source[%d]: duplicate key %q", i, src.Key)
		}
		keys[src.Key] = struct{}{}

		if strings.TrimSpace(src.SeedURL) == "" {
			return fmt.Errorf("source %q: seed_url is required", src.Key)
		}
		u, err := url.Parse(src.SeedURL)
		if err != nil || u.Scheme == "" || u.Host == "" {
			return fmt.Errorf("source %q: seed_url %q is not a valid absolute URL", src.Key, src.SeedURL)
		}

		for j, tag := range src.Tags {
			trimmed := strings.TrimSpace(tag)
			if trimmed == "" {
				return fmt.Errorf("source %q: tag[%d] is empty", src.Key, j)
			}
			sources[i].Tags[j] = trimmed
		}
		// Normalise empty slice to nil so round-trip equality (nil vs []) is
		// stable across a load → write → load round trip.
		if len(sources[i].Tags) == 0 {
			sources[i].Tags = nil
		}
	}
	return nil
}
