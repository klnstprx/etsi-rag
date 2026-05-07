package config_test

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"etsi-rag/internal/config"
	"etsi-rag/internal/domain"
)

const validToml = `
[[sources]]
key = 'a'
name = 'Source A'
seed_url = 'https://example.com/a/'
description = 'Source A description'
tags = ['alpha', 'beta']
enabled = true

[[sources]]
key = 'b'
name = 'Source B'
seed_url = 'https://example.com/b/'
enabled = false
`

func writeTempSources(t *testing.T, contents string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "sources.toml")
	if err := os.WriteFile(path, []byte(contents), 0o644); err != nil {
		t.Fatalf("writing temp sources: %v", err)
	}
	return path
}

func TestLoadSourcesFile_Valid(t *testing.T) {
	path := writeTempSources(t, validToml)

	sources, err := config.LoadSourcesFile(path)
	if err != nil {
		t.Fatalf("LoadSourcesFile: %v", err)
	}
	if len(sources) != 2 {
		t.Fatalf("want 2 sources, got %d", len(sources))
	}

	want := domain.CrawlSource{
		Key:         "a",
		Name:        "Source A",
		SeedURL:     "https://example.com/a/",
		Description: "Source A description",
		Tags:        []string{"alpha", "beta"},
		Enabled:     true,
	}
	if !reflect.DeepEqual(sources[0], want) {
		t.Errorf("sources[0] = %+v, want %+v", sources[0], want)
	}
	if sources[1].Key != "b" || sources[1].Enabled {
		t.Errorf("sources[1] mismatch: %+v", sources[1])
	}
	if len(sources[1].Tags) != 0 {
		t.Errorf("sources[1].Tags = %v, want empty", sources[1].Tags)
	}
}

func TestLoadSourcesFile_RejectsDuplicateKey(t *testing.T) {
	dup := `
[[sources]]
key = 'a'
name = 'A1'
seed_url = 'https://example.com/a1/'

[[sources]]
key = 'a'
name = 'A2'
seed_url = 'https://example.com/a2/'
`
	path := writeTempSources(t, dup)

	if _, err := config.LoadSourcesFile(path); err == nil {
		t.Fatal("expected duplicate-key error, got nil")
	}
}

func TestLoadSourcesFile_RejectsEmptyKey(t *testing.T) {
	bad := `
[[sources]]
name = 'No Key'
seed_url = 'https://example.com/'
`
	path := writeTempSources(t, bad)

	if _, err := config.LoadSourcesFile(path); err == nil {
		t.Fatal("expected missing-key error, got nil")
	}
}

func TestLoadSourcesFile_RejectsInvalidSeedURL(t *testing.T) {
	bad := `
[[sources]]
key = 'a'
name = 'A'
seed_url = 'not-a-url'
`
	path := writeTempSources(t, bad)

	if _, err := config.LoadSourcesFile(path); err == nil {
		t.Fatal("expected invalid-url error, got nil")
	}
}

func TestWriteSources_RoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "sources.toml")

	in := []domain.CrawlSource{
		{Key: "x", Name: "X", SeedURL: "https://example.com/x/", Description: "Hub X", Tags: []string{"foo", "bar"}, Enabled: true},
		{Key: "y", Name: "Y", SeedURL: "https://example.com/y/", Enabled: false},
	}
	if err := config.WriteSources(in, path); err != nil {
		t.Fatalf("WriteSources: %v", err)
	}

	out, err := config.LoadSourcesFile(path)
	if err != nil {
		t.Fatalf("LoadSourcesFile after write: %v", err)
	}
	if len(out) != len(in) {
		t.Fatalf("round-trip length: got %d, want %d", len(out), len(in))
	}
	for i, s := range out {
		if !reflect.DeepEqual(s, in[i]) {
			t.Errorf("source[%d] round-trip: got %+v, want %+v", i, s, in[i])
		}
	}
}

func TestLoadSourcesFile_RejectsEmptyTag(t *testing.T) {
	bad := `
[[sources]]
key = 'a'
name = 'A'
seed_url = 'https://example.com/'
tags = ['valid', '   ']
`
	path := writeTempSources(t, bad)
	if _, err := config.LoadSourcesFile(path); err == nil {
		t.Fatal("expected empty-tag error, got nil")
	}
}

func TestWriteSources_RejectsInvalid(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "sources.toml")

	bad := []domain.CrawlSource{
		{Key: "", Name: "no key", SeedURL: "https://example.com/"},
	}
	if err := config.WriteSources(bad, path); err == nil {
		t.Fatal("expected validation error on empty key")
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("file should not have been created: stat err=%v", err)
	}
}

func TestEnsureSourcesFile_CopiesDefault(t *testing.T) {
	dir := t.TempDir()
	defaultPath := filepath.Join(dir, config.DefaultSourcesFileName)
	if err := os.WriteFile(defaultPath, []byte(validToml), 0o644); err != nil {
		t.Fatalf("seeding default: %v", err)
	}

	if err := config.EnsureSourcesFile(dir); err != nil {
		t.Fatalf("EnsureSourcesFile: %v", err)
	}

	userPath := filepath.Join(dir, config.SourcesFileName)
	if _, err := os.Stat(userPath); err != nil {
		t.Fatalf("user sources file missing after ensure: %v", err)
	}

	// Second call must be a no-op (does not overwrite user edits).
	if err := os.WriteFile(userPath, []byte("# user edit\n"), 0o644); err != nil {
		t.Fatalf("user edit: %v", err)
	}
	if err := config.EnsureSourcesFile(dir); err != nil {
		t.Fatalf("EnsureSourcesFile second call: %v", err)
	}
	got, _ := os.ReadFile(userPath)
	if string(got) != "# user edit\n" {
		t.Errorf("EnsureSourcesFile clobbered user edit; got %q", got)
	}
}
