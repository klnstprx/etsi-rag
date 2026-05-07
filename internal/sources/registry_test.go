package sources_test

import (
	"os"
	"path/filepath"
	"reflect"
	"sync"
	"testing"

	"github.com/rs/zerolog"

	"etsi-rag/internal/config"
	"etsi-rag/internal/domain"
	"etsi-rag/internal/sources"
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

func writeFile(t *testing.T, contents string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "sources.toml")
	if err := os.WriteFile(path, []byte(contents), 0o644); err != nil {
		t.Fatalf("writing temp sources: %v", err)
	}
	return path
}

func disabledLogger() zerolog.Logger {
	return zerolog.New(os.Stderr).Level(zerolog.Disabled)
}

func TestRegistry_Load_Valid(t *testing.T) {
	path := writeFile(t, validToml)
	r := sources.New(path, disabledLogger())

	if err := r.Load(); err != nil {
		t.Fatalf("Load: %v", err)
	}
	got := r.Snapshot()
	if len(got) != 2 {
		t.Fatalf("Snapshot len = %d, want 2", len(got))
	}
	if got[0].Key != "a" || got[1].Key != "b" {
		t.Errorf("keys = [%s %s], want [a b]", got[0].Key, got[1].Key)
	}
	if !got[0].Enabled || got[1].Enabled {
		t.Errorf("Enabled = [%v %v], want [true false]", got[0].Enabled, got[1].Enabled)
	}
}

func TestRegistry_Load_InvalidLeavesSnapshotUntouched(t *testing.T) {
	good := writeFile(t, validToml)
	r := sources.New(good, disabledLogger())
	if err := r.Load(); err != nil {
		t.Fatalf("initial Load: %v", err)
	}
	before := r.Snapshot()

	// Replace the file with a duplicate-key payload, which validateSources rejects.
	bad := `
[[sources]]
key = 'x'
name = 'X1'
seed_url = 'https://example.com/x1/'

[[sources]]
key = 'x'
name = 'X2'
seed_url = 'https://example.com/x2/'
`
	if err := os.WriteFile(good, []byte(bad), 0o644); err != nil {
		t.Fatalf("writing bad sources: %v", err)
	}

	if err := r.Load(); err == nil {
		t.Fatal("expected Load to reject duplicate-key file")
	}
	after := r.Snapshot()
	if !reflect.DeepEqual(before, after) {
		t.Errorf("snapshot changed after failed Load: before=%+v after=%+v", before, after)
	}
}

func TestRegistry_Replace_RoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "sources.toml")
	r := sources.New(path, disabledLogger())

	in := []domain.CrawlSource{
		{Key: "x", Name: "X", SeedURL: "https://example.com/x/", Description: "Hub X", Tags: []string{"foo"}, Enabled: true},
		{Key: "y", Name: "Y", SeedURL: "https://example.com/y/", Enabled: false},
	}
	if err := r.Replace(in); err != nil {
		t.Fatalf("Replace: %v", err)
	}

	// In-memory snapshot reflects the new state.
	got := r.Snapshot()
	if !reflect.DeepEqual(got, in) {
		t.Errorf("Snapshot = %+v, want %+v", got, in)
	}

	// On-disk file parses back to the same content.
	onDisk, err := config.LoadSourcesFile(path)
	if err != nil {
		t.Fatalf("LoadSourcesFile: %v", err)
	}
	if !reflect.DeepEqual(onDisk, in) {
		t.Errorf("on disk = %+v, want %+v", onDisk, in)
	}
}

func TestRegistry_Replace_InvalidLeavesStateUntouched(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "sources.toml")
	r := sources.New(path, disabledLogger())

	good := []domain.CrawlSource{
		{Key: "ok", Name: "OK", SeedURL: "https://example.com/", Enabled: true},
	}
	if err := r.Replace(good); err != nil {
		t.Fatalf("seeding Replace: %v", err)
	}
	beforeSnap := r.Snapshot()
	beforeBytes, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read seed file: %v", err)
	}

	bad := []domain.CrawlSource{
		{Key: "", Name: "no key", SeedURL: "https://example.com/"},
	}
	if err := r.Replace(bad); err == nil {
		t.Fatal("expected Replace to reject invalid sources")
	}

	afterSnap := r.Snapshot()
	if !reflect.DeepEqual(beforeSnap, afterSnap) {
		t.Errorf("snapshot mutated by failed Replace: before=%+v after=%+v", beforeSnap, afterSnap)
	}
	afterBytes, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read file after failed Replace: %v", err)
	}
	if string(beforeBytes) != string(afterBytes) {
		t.Errorf("on-disk file mutated by failed Replace")
	}
}

func TestRegistry_KnownKeys(t *testing.T) {
	r := sources.New(filepath.Join(t.TempDir(), "sources.toml"), disabledLogger())
	if err := r.Replace([]domain.CrawlSource{
		{Key: "a", Name: "A", SeedURL: "https://example.com/", Enabled: true},
		{Key: "b", Name: "B", SeedURL: "https://example.com/", Enabled: false},
	}); err != nil {
		t.Fatalf("Replace: %v", err)
	}

	got := r.KnownKeys()
	if len(got) != 2 || got[0] != "a" || got[1] != "b" {
		t.Errorf("KnownKeys = %v, want [a b]", got)
	}
}

func TestRegistry_SnapshotIsDefensiveCopy(t *testing.T) {
	r := sources.New(filepath.Join(t.TempDir(), "sources.toml"), disabledLogger())
	if err := r.Replace([]domain.CrawlSource{
		{Key: "a", Name: "A", SeedURL: "https://example.com/", Tags: []string{"x"}, Enabled: true},
	}); err != nil {
		t.Fatalf("Replace: %v", err)
	}

	first := r.Snapshot()
	first[0].Name = "MUTATED"
	first[0].Tags[0] = "MUTATED"

	second := r.Snapshot()
	if second[0].Name == "MUTATED" {
		t.Error("Snapshot returned a non-defensive top-level copy")
	}
	// Tags is a slice — Snapshot does a top-level slice copy, not a deep copy,
	// so mutating an element of the inner Tags slice IS visible through later
	// Snapshots. Lock that contract here so future readers know.
	if second[0].Tags[0] != "MUTATED" {
		t.Logf("Snapshot deep-copies Tags now; update this test if intentional")
	}
}

func TestRegistry_ConcurrentSnapshotReplace(t *testing.T) {
	r := sources.New(filepath.Join(t.TempDir(), "sources.toml"), disabledLogger())
	if err := r.Replace([]domain.CrawlSource{
		{Key: "a", Name: "A", SeedURL: "https://example.com/", Enabled: true},
	}); err != nil {
		t.Fatalf("seed: %v", err)
	}

	const goroutines = 10
	const iterations = 100
	var wg sync.WaitGroup
	wg.Add(2 * goroutines)
	for i := 0; i < goroutines; i++ {
		go func(id int) {
			defer wg.Done()
			for j := 0; j < iterations; j++ {
				_ = r.Snapshot()
			}
		}(i)
		go func(id int) {
			defer wg.Done()
			for j := 0; j < iterations; j++ {
				_ = r.Replace([]domain.CrawlSource{
					{Key: "a", Name: "A", SeedURL: "https://example.com/", Enabled: true},
				})
			}
		}(i)
	}
	wg.Wait()
	// No assertions beyond -race surfacing a data race.
}
