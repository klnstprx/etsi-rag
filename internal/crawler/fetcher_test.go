package crawler

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestFetch_HTMLSuccess(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("User-Agent") != userAgent {
			t.Errorf("expected User-Agent %q, got %q", userAgent, r.Header.Get("User-Agent"))
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Write([]byte("<html><body>Hello</body></html>"))
	}))
	defer srv.Close()

	f := NewFetcher(5*time.Second, 0, 0)
	result, err := f.Fetch(context.Background(), srv.URL+"/page")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.ContentType != "html" {
		t.Errorf("expected content type html, got %s", result.ContentType)
	}
	if result.StatusCode != 200 {
		t.Errorf("expected status 200, got %d", result.StatusCode)
	}
	if !strings.Contains(string(result.Body), "Hello") {
		t.Errorf("expected body to contain 'Hello', got %s", string(result.Body))
	}
}

func TestFetch_PDFDetection(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/pdf")
		w.Write([]byte("%PDF-1.4 fake content"))
	}))
	defer srv.Close()

	f := NewFetcher(5*time.Second, 0, 0)
	result, err := f.Fetch(context.Background(), srv.URL+"/doc.pdf")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.ContentType != "pdf" {
		t.Errorf("expected content type pdf, got %s", result.ContentType)
	}
}

func TestFetch_PDFDetectionByURL(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// No Content-Type header hint, but URL ends in .pdf
		w.Write([]byte("%PDF-1.4 fake content"))
	}))
	defer srv.Close()

	f := NewFetcher(5*time.Second, 0, 0)
	result, err := f.Fetch(context.Background(), srv.URL+"/document.pdf")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if result.ContentType != "pdf" {
		t.Errorf("expected content type pdf from URL suffix, got %s", result.ContentType)
	}
}

func TestFetch_BodySizeLimit(t *testing.T) {
	largeBody := strings.Repeat("x", 1024)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(largeBody))
	}))
	defer srv.Close()

	// Limit to 512 bytes
	f := NewFetcher(5*time.Second, 0, 512)
	_, err := f.Fetch(context.Background(), srv.URL+"/big")
	if err == nil {
		t.Fatal("expected error for body exceeding size limit")
	}
	if !strings.Contains(err.Error(), "exceeds") {
		t.Errorf("expected 'exceeds' in error message, got: %v", err)
	}
}

func TestFetch_BodySizeLimitZeroUnlimited(t *testing.T) {
	body := strings.Repeat("x", 4096)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(body))
	}))
	defer srv.Close()

	// 0 = unlimited
	f := NewFetcher(5*time.Second, 0, 0)
	result, err := f.Fetch(context.Background(), srv.URL+"/big")
	if err != nil {
		t.Fatalf("unexpected error with unlimited body size: %v", err)
	}
	if len(result.Body) != 4096 {
		t.Errorf("expected 4096 bytes, got %d", len(result.Body))
	}
}

func TestFetch_StatusError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
	}))
	defer srv.Close()

	f := NewFetcher(5*time.Second, 0, 0)
	_, err := f.Fetch(context.Background(), srv.URL+"/missing")
	if err == nil {
		t.Fatal("expected error for 404 status")
	}
	if !strings.Contains(err.Error(), "404") {
		t.Errorf("expected '404' in error message, got: %v", err)
	}
}

func TestFetch_ContextCancelled(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(5 * time.Second)
		w.Write([]byte("slow"))
	}))
	defer srv.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately

	f := NewFetcher(5*time.Second, 0, 0)
	_, err := f.Fetch(ctx, srv.URL+"/slow")
	if err == nil {
		t.Fatal("expected error for cancelled context")
	}
}

func TestFetch_BodySizeLimitExact(t *testing.T) {
	// Body exactly at limit should succeed
	body := strings.Repeat("x", 512)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte(body))
	}))
	defer srv.Close()

	f := NewFetcher(5*time.Second, 0, 512)
	result, err := f.Fetch(context.Background(), srv.URL+"/exact")
	if err != nil {
		t.Fatalf("body at exact limit should succeed: %v", err)
	}
	if len(result.Body) != 512 {
		t.Errorf("expected 512 bytes, got %d", len(result.Body))
	}
}
