package objectstore

import (
	"context"
	"encoding/xml"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"etsi-rag/internal/config"
)

// fakeS3 is a minimal in-memory S3 backend for testing.
type fakeS3 struct {
	mu      sync.RWMutex
	objects map[string][]byte
}

func newFakeS3() *fakeS3 {
	return &fakeS3{objects: make(map[string][]byte)}
}

// listResult mirrors the S3 ListObjectsV2 XML response.
type listResult struct {
	XMLName  xml.Name     `xml:"ListBucketResult"`
	Contents []listObject `xml:"Contents"`
}

type listObject struct {
	Key string `xml:"Key"`
}

func (f *fakeS3) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// Path format: /{bucket}/{key...}
	parts := strings.SplitN(strings.TrimPrefix(r.URL.Path, "/"), "/", 2)
	bucket := parts[0]
	key := ""
	if len(parts) > 1 {
		key = parts[1]
	}

	switch r.Method {
	case http.MethodPut:
		data, _ := io.ReadAll(r.Body)
		f.mu.Lock()
		f.objects[key] = data
		f.mu.Unlock()
		w.WriteHeader(http.StatusOK)

	case http.MethodGet:
		// ListObjectsV2: GET /{bucket}?list-type=2&prefix=...
		if key == "" && r.URL.Query().Get("list-type") == "2" {
			prefix := r.URL.Query().Get("prefix")
			f.mu.RLock()
			var items []listObject
			for k := range f.objects {
				if strings.HasPrefix(k, prefix) {
					items = append(items, listObject{Key: k})
				}
			}
			f.mu.RUnlock()

			w.Header().Set("Content-Type", "application/xml")
			xml.NewEncoder(w).Encode(listResult{Contents: items})
			return
		}

		f.mu.RLock()
		data, ok := f.objects[key]
		f.mu.RUnlock()
		if !ok {
			http.Error(w, "NoSuchKey", http.StatusNotFound)
			return
		}
		w.Write(data)

	case http.MethodDelete:
		f.mu.Lock()
		delete(f.objects, key)
		f.mu.Unlock()
		w.WriteHeader(http.StatusNoContent)

	case http.MethodHead:
		if key == "" {
			// HeadBucket
			_ = bucket
			w.WriteHeader(http.StatusOK)
			return
		}
		f.mu.RLock()
		_, ok := f.objects[key]
		f.mu.RUnlock()
		if !ok {
			http.Error(w, "NoSuchKey", http.StatusNotFound)
			return
		}
		w.WriteHeader(http.StatusOK)

	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}

func newTestClient(t *testing.T) (*Client, *httptest.Server) {
	t.Helper()
	fake := newFakeS3()
	srv := httptest.NewServer(fake)
	t.Cleanup(srv.Close)

	client, err := New(config.ObjectStoreConfig{
		Endpoint:  srv.URL,
		Bucket:    "test-bucket",
		AccessKey: "test",
		SecretKey: "test",
		Region:    "us-east-1",
	})
	if err != nil {
		t.Fatal(err)
	}
	return client, srv
}

func TestNew_EmptyEndpoint(t *testing.T) {
	_, err := New(config.ObjectStoreConfig{Endpoint: ""})
	if err == nil {
		t.Fatal("expected error for empty endpoint")
	}
}

func TestNew_ValidConfig(t *testing.T) {
	fake := newFakeS3()
	srv := httptest.NewServer(fake)
	defer srv.Close()

	client, err := New(config.ObjectStoreConfig{
		Endpoint:  srv.URL,
		Bucket:    "b",
		AccessKey: "a",
		SecretKey: "s",
		Region:    "us-east-1",
	})
	if err != nil {
		t.Fatal(err)
	}
	if client == nil {
		t.Fatal("expected non-nil client")
	}
}

func TestPutAndGet(t *testing.T) {
	c, _ := newTestClient(t)
	ctx := context.Background()

	data := []byte("hello world")
	if err := c.Put(ctx, "test/key.txt", data, "text/plain"); err != nil {
		t.Fatal(err)
	}

	got, err := c.Get(ctx, "test/key.txt")
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(data) {
		t.Errorf("got %q, want %q", got, data)
	}
}

func TestGet_NotFound(t *testing.T) {
	c, _ := newTestClient(t)
	ctx := context.Background()

	_, err := c.Get(ctx, "nonexistent")
	if err == nil {
		t.Fatal("expected error for missing key")
	}
}

func TestDelete(t *testing.T) {
	c, _ := newTestClient(t)
	ctx := context.Background()

	c.Put(ctx, "del-key", []byte("data"), "application/octet-stream")
	if err := c.Delete(ctx, "del-key"); err != nil {
		t.Fatal(err)
	}

	_, err := c.Get(ctx, "del-key")
	if err == nil {
		t.Fatal("expected error after delete")
	}
}

func TestList(t *testing.T) {
	c, _ := newTestClient(t)
	ctx := context.Background()

	// 3 matching keys + 1 non-matching.
	for _, key := range []string{"docs/a", "docs/b", "docs/c", "other/x"} {
		c.Put(ctx, key, []byte("data"), "text/plain")
	}

	keys, err := c.List(ctx, "docs/")
	if err != nil {
		t.Fatal(err)
	}
	if len(keys) != 3 {
		t.Errorf("got %d keys, want 3", len(keys))
	}

	for _, k := range keys {
		if !strings.HasPrefix(k, "docs/") {
			t.Errorf("unexpected key %q in results", k)
		}
	}
}

func TestList_Empty(t *testing.T) {
	c, _ := newTestClient(t)
	ctx := context.Background()

	keys, err := c.List(ctx, "nonexistent/")
	if err != nil {
		t.Fatal(err)
	}
	if len(keys) != 0 {
		t.Errorf("got %d keys, want 0", len(keys))
	}
}

func TestHealthy(t *testing.T) {
	c, _ := newTestClient(t)
	ctx := context.Background()

	if !c.Healthy(ctx) {
		t.Error("expected Healthy to return true for working server")
	}
}

func TestHealthy_Down(t *testing.T) {
	c, srv := newTestClient(t)

	srv.Close()

	// Short timeout — the server is closed so it should fail fast,
	// but the AWS SDK retries with backoff without a deadline.
	ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancel()

	if c.Healthy(ctx) {
		t.Error("expected Healthy to return false for closed server")
	}
}

func TestExists(t *testing.T) {
	c, _ := newTestClient(t)
	ctx := context.Background()

	c.Put(ctx, "exists-key", []byte("data"), "text/plain")

	exists, err := c.Exists(ctx, "exists-key")
	if err != nil {
		t.Fatal(err)
	}
	if !exists {
		t.Error("expected Exists to return true for existing key")
	}

	exists, err = c.Exists(ctx, "missing-key")
	if err != nil {
		t.Fatal(err)
	}
	if exists {
		t.Error("expected Exists to return false for missing key")
	}
}
