package objectstore

import (
	"context"
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

func (f *fakeS3) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	// Path format: /{bucket}/{key...}
	parts := strings.SplitN(strings.TrimPrefix(r.URL.Path, "/"), "/", 2)
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
		f.mu.RLock()
		data, ok := f.objects[key]
		f.mu.RUnlock()
		if !ok {
			http.Error(w, "NoSuchKey", http.StatusNotFound)
			return
		}
		w.Write(data)

	case http.MethodHead: // HeadBucket; the client never heads objects.
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

func TestPutAndGet_StalledStoreTimesOut(t *testing.T) {
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case <-release:
		case <-r.Context().Done():
		}
	}))
	defer srv.Close()
	defer close(release) // runs before srv.Close, which waits for handlers

	c, err := New(config.ObjectStoreConfig{
		Endpoint: srv.URL, Bucket: "b", AccessKey: "a", SecretKey: "s", Region: "us-east-1",
	})
	if err != nil {
		t.Fatal(err)
	}
	c.timeout = 100 * time.Millisecond

	ops := map[string]func() error{
		"put": func() error { return c.Put(context.Background(), "k", []byte("data"), "text/plain") },
		"get": func() error { _, err := c.Get(context.Background(), "k"); return err },
	}
	for name, op := range ops {
		t.Run(name, func(t *testing.T) {
			done := make(chan error, 1)
			go func() { done <- op() }()
			select {
			case err := <-done:
				if err == nil {
					t.Fatal("expected an error from a stalled store")
				}
			case <-time.After(5 * time.Second):
				t.Fatal("still blocked 5s into a stalled request, want it to give up after the timeout")
			}
		})
	}
}
