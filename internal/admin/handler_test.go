package admin

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/rs/zerolog"

	"etsi-rag/internal/agent"
	"etsi-rag/internal/config"
	"etsi-rag/internal/domain"
	"etsi-rag/internal/prompts"
	"etsi-rag/internal/session"
	"etsi-rag/internal/store"
)

func setupTestAdminHandler(t *testing.T) (*Handler, *store.SQLiteStore) {
	t.Helper()

	db, err := store.OpenDB(":memory:")
	if err != nil {
		t.Fatal(err)
	}
	if err := store.Migrate(db); err != nil {
		db.Close()
		t.Fatal(err)
	}

	s := store.NewSQLiteStore(db, func() int { return 4 })
	t.Cleanup(func() { s.Close() })

	cfg := &config.Config{}
	cfg.Server.Host = "localhost"
	cfg.Server.Port = 8080
	// Validate-on-save needs cross-section invariants to hold, so seed
	// defaults that match Load()'s applyDefaults output.
	cfg.Retrieval.MaxContextTokens = 4096
	cfg.Retrieval.RetrievalMode = "dense"

	sessionStore := session.NewStore(config.SessionConfig{MaxMessages: 10, TTLSeconds: 3600})
	t.Cleanup(sessionStore.Close)

	tmpDir := t.TempDir()
	configPath := tmpDir + "/config.toml"

	h := NewHandler(HandlerConfig{
		Store:        s,
		Config:       cfg,
		ConfigPath:   configPath,
		SessionStore: sessionStore,
		Logger:       zerolog.New(os.Stderr).Level(zerolog.Disabled),
	})

	return h, s
}

func withChiParam(r *http.Request, key, value string) *http.Request {
	rctx := chi.NewRouteContext()
	rctx.URLParams.Add(key, value)
	return r.WithContext(context.WithValue(r.Context(), chi.RouteCtxKey, rctx))
}

func postForm(target string, values url.Values) *http.Request {
	req := httptest.NewRequest(http.MethodPost, target, strings.NewReader(values.Encode()))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	return req
}

func putForm(target string, values url.Values) *http.Request {
	req := httptest.NewRequest(http.MethodPut, target, strings.NewReader(values.Encode()))
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	return req
}

// --- formInt / formFloat ---

func TestFormInt(t *testing.T) {
	tests := []struct {
		name     string
		value    string
		fallback int
		want     int
	}{
		{"valid", "42", 0, 42},
		{"empty uses fallback", "", 99, 99},
		{"non-numeric uses fallback", "abc", 99, 99},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			form := url.Values{"key": {tt.value}}
			req := httptest.NewRequest(http.MethodGet, "/?"+form.Encode(), nil)
			got := formInt(req, "key", tt.fallback)
			if got != tt.want {
				t.Errorf("formInt() = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestFormFloat(t *testing.T) {
	tests := []struct {
		name     string
		value    string
		fallback float64
		want     float64
	}{
		{"valid", "3.14", 0, 3.14},
		{"empty uses fallback", "", 1.5, 1.5},
		{"non-numeric uses fallback", "abc", 1.5, 1.5},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			form := url.Values{"key": {tt.value}}
			req := httptest.NewRequest(http.MethodGet, "/?"+form.Encode(), nil)
			got := formFloat(req, "key", tt.fallback)
			if got != tt.want {
				t.Errorf("formFloat() = %f, want %f", got, tt.want)
			}
		})
	}
}

// --- isHTMX ---

func TestIsHTMX(t *testing.T) {
	tests := []struct {
		name      string
		hxRequest string
		hxBoosted string
		want      bool
	}{
		{"HTMX request", "true", "", true},
		{"boosted is not HTMX fragment", "true", "true", false},
		{"neither", "", "", false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodGet, "/", nil)
			if tt.hxRequest != "" {
				req.Header.Set("HX-Request", tt.hxRequest)
			}
			if tt.hxBoosted != "" {
				req.Header.Set("HX-Boosted", tt.hxBoosted)
			}
			if got := isHTMX(req); got != tt.want {
				t.Errorf("isHTMX() = %v, want %v", got, tt.want)
			}
		})
	}
}

// --- ConfigPage (GET) ---

func TestConfigPage_DefaultsToFirstSection(t *testing.T) {
	h, _ := setupTestAdminHandler(t)

	req := httptest.NewRequest(http.MethodGet, "/admin/config", nil)
	w := httptest.NewRecorder()

	h.ConfigPage(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	body := w.Body.String()
	// First section in configToSections is "server" — its slug should appear
	// as the active section's wrapping div id.
	if !strings.Contains(body, `id="config-section-server"`) {
		t.Errorf("default render should include the server section; body did not contain its id")
	}
	// Submenu should render group headings.
	if !strings.Contains(body, "Infrastructure") {
		t.Error("submenu should show the Infrastructure group heading")
	}
}

func TestConfigPage_ShowsSelectedSection(t *testing.T) {
	h, _ := setupTestAdminHandler(t)

	req := httptest.NewRequest(http.MethodGet, "/admin/config/retrieval", nil)
	req = withChiParam(req, "section", "retrieval")
	w := httptest.NewRecorder()

	h.ConfigPage(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	body := w.Body.String()
	if !strings.Contains(body, `id="config-section-retrieval"`) {
		t.Errorf("expected retrieval section to render; body missing the id")
	}
	// The retrieval submenu link should carry aria-current="page" on its
	// active state.
	if !strings.Contains(body, `aria-current="page"`) {
		t.Error("active submenu link should carry aria-current=page")
	}
}

func TestConfigPage_UnknownSection404(t *testing.T) {
	h, _ := setupTestAdminHandler(t)

	req := httptest.NewRequest(http.MethodGet, "/admin/config/bogus", nil)
	req = withChiParam(req, "section", "bogus")
	w := httptest.NewRecorder()

	h.ConfigPage(w, req)

	if w.Code != http.StatusNotFound {
		t.Errorf("status = %d, want 404 for unknown section", w.Code)
	}
}

// --- ConfigSave ---

func TestConfigSave_Server(t *testing.T) {
	h, _ := setupTestAdminHandler(t)

	form := url.Values{
		"host":          {"0.0.0.0"},
		"port":          {"9090"},
		"read_timeout":  {"30"},
		"write_timeout": {"60"},
	}
	req := postForm("/admin/config/server", form)
	req = withChiParam(req, "section", "server")
	w := httptest.NewRecorder()

	h.ConfigSave(w, req)

	if h.config.Server.Host != "0.0.0.0" {
		t.Errorf("Host = %q, want %q", h.config.Server.Host, "0.0.0.0")
	}
	if h.config.Server.Port != 9090 {
		t.Errorf("Port = %d, want 9090", h.config.Server.Port)
	}
	if h.config.Server.ReadTimeout != 30 {
		t.Errorf("ReadTimeout = %d, want 30", h.config.Server.ReadTimeout)
	}
	if h.config.Server.WriteTimeout != 60 {
		t.Errorf("WriteTimeout = %d, want 60", h.config.Server.WriteTimeout)
	}
}

func TestConfigSave_Retrieval(t *testing.T) {
	h, _ := setupTestAdminHandler(t)

	form := url.Values{
		"top_k":              {"10"},
		"max_context_tokens": {"4096"},
	}
	req := postForm("/admin/config/retrieval", form)
	req = withChiParam(req, "section", "retrieval")
	w := httptest.NewRecorder()

	h.ConfigSave(w, req)

	if h.config.Retrieval.TopK != 10 {
		t.Errorf("TopK = %d, want 10", h.config.Retrieval.TopK)
	}
	if h.config.Retrieval.MaxContextTokens != 4096 {
		t.Errorf("MaxContextTokens = %d, want 4096", h.config.Retrieval.MaxContextTokens)
	}
}

func TestConfigSave_Chunking(t *testing.T) {
	h, _ := setupTestAdminHandler(t)

	form := url.Values{
		"max_chunk_size": {"512"},
		"chunk_overlap":  {"64"},
		"min_chunk_size": {"128"},
	}
	req := postForm("/admin/config/chunking", form)
	req = withChiParam(req, "section", "chunking")
	w := httptest.NewRecorder()

	h.ConfigSave(w, req)

	if h.config.Chunking.MaxChunkSize != 512 {
		t.Errorf("MaxChunkSize = %d, want 512", h.config.Chunking.MaxChunkSize)
	}
	if h.config.Chunking.ChunkOverlap != 64 {
		t.Errorf("ChunkOverlap = %d, want 64", h.config.Chunking.ChunkOverlap)
	}
	if h.config.Chunking.MinChunkSize != 128 {
		t.Errorf("MinChunkSize = %d, want 128", h.config.Chunking.MinChunkSize)
	}
}

// TestConfigSave_RejectsInvalidConfig posts a chunking section with
// max_chunk_size larger than retrieval.max_context_tokens. Validate should
// reject the save: the in-memory config reverts, no callback fires, and the
// HTML response includes an error toast.
func TestConfigSave_RejectsInvalidConfig(t *testing.T) {
	h, _ := setupTestAdminHandler(t)
	h.config.Retrieval.MaxContextTokens = 1024
	originalChunkSize := h.config.Chunking.MaxChunkSize

	fired := false
	h.SetOnConfigSave(func() { fired = true })

	form := url.Values{
		"max_chunk_size": {"99999"}, // larger than retrieval.max_context_tokens
		"chunk_overlap":  {"32"},
		"min_chunk_size": {"64"},
	}
	req := postForm("/admin/config/chunking", form)
	req = withChiParam(req, "section", "chunking")
	w := httptest.NewRecorder()

	h.ConfigSave(w, req)

	if h.config.Chunking.MaxChunkSize != originalChunkSize {
		t.Errorf("MaxChunkSize = %d, want %d (revert on validation failure)",
			h.config.Chunking.MaxChunkSize, originalChunkSize)
	}
	if fired {
		t.Error("onConfigSave fired despite validation failure")
	}
	body := w.Body.String()
	if !strings.Contains(body, "Invalid config") {
		t.Errorf("response body missing error toast; got:\n%s", body)
	}
	if !strings.Contains(body, "max_context_tokens") {
		t.Errorf("error toast should mention the violated invariant; got:\n%s", body)
	}
}

func TestConfigSave_UnknownSection(t *testing.T) {
	h, _ := setupTestAdminHandler(t)

	req := postForm("/admin/config/bogus", url.Values{})
	req = withChiParam(req, "section", "bogus")
	w := httptest.NewRecorder()

	h.ConfigSave(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", w.Code)
	}
}

func TestConfigSave_FiresCallback(t *testing.T) {
	h, _ := setupTestAdminHandler(t)

	fired := false
	h.SetOnConfigSave(func() { fired = true })

	form := url.Values{"host": {"localhost"}, "port": {"8080"}}
	req := postForm("/admin/config/server", form)
	req = withChiParam(req, "section", "server")
	w := httptest.NewRecorder()

	h.ConfigSave(w, req)

	if !fired {
		t.Error("onConfigSave callback was not fired")
	}
}

func TestConfigSave_Agent_CustomPrompt(t *testing.T) {
	h, _ := setupTestAdminHandler(t)

	custom := "You are a custom bot for testing."
	form := url.Values{
		"backend":        {""},
		"model":          {""},
		"max_tokens":     {"1024"},
		"temperature":    {"0.5"},
		"context_window": {"8192"},
		"max_iterations": {"10"},
		"system_prompt":  {custom},
	}
	req := postForm("/admin/config/agent", form)
	req = withChiParam(req, "section", "agent")
	w := httptest.NewRecorder()

	h.ConfigSave(w, req)

	if h.config.Agent.SystemPrompt != custom {
		t.Errorf("SystemPrompt = %q, want %q", h.config.Agent.SystemPrompt, custom)
	}
}

func TestConfigSave_Agent_ClearPrompt(t *testing.T) {
	h, _ := setupTestAdminHandler(t)
	h.config.Agent.SystemPrompt = "some custom prompt"

	form := url.Values{
		"backend":        {""},
		"model":          {""},
		"max_tokens":     {"1024"},
		"temperature":    {"0.5"},
		"context_window": {"8192"},
		"max_iterations": {"10"},
		"system_prompt":  {""},
	}
	req := postForm("/admin/config/agent", form)
	req = withChiParam(req, "section", "agent")
	w := httptest.NewRecorder()

	h.ConfigSave(w, req)

	if h.config.Agent.SystemPrompt != "" {
		t.Errorf("SystemPrompt = %q, want empty (default)", h.config.Agent.SystemPrompt)
	}
}

func TestConfigSave_Agent_DefaultTextNormalized(t *testing.T) {
	h, _ := setupTestAdminHandler(t)

	// Submitting the default template text should normalize to empty string
	// so config stays clean and auto-updates with future default changes.
	form := url.Values{
		"backend":        {""},
		"model":          {""},
		"max_tokens":     {"1024"},
		"temperature":    {"0.5"},
		"context_window": {"8192"},
		"max_iterations": {"10"},
		"system_prompt":  {prompts.Agent},
	}
	req := postForm("/admin/config/agent", form)
	req = withChiParam(req, "section", "agent")
	w := httptest.NewRecorder()

	h.ConfigSave(w, req)

	if h.config.Agent.SystemPrompt != "" {
		t.Errorf("SystemPrompt = %q, want empty (default text should normalize)", h.config.Agent.SystemPrompt)
	}
}

func TestConfigSave_EmbeddingDimensionChange_WarnsReprocess(t *testing.T) {
	h, _ := setupTestAdminHandler(t)
	h.config.Embedding.Dimension = 512 // initial

	form := url.Values{
		"backend":    {""},
		"model":      {""},
		"dimension":  {"1024"}, // changed
		"max_tokens": {"512"},
		"batch_size": {"32"},
	}
	req := postForm("/admin/config/embedding", form)
	req = withChiParam(req, "section", "embedding")
	w := httptest.NewRecorder()

	h.ConfigSave(w, req)

	body := w.Body.String()
	if !strings.Contains(body, "existing documents should be reprocessed") {
		t.Error("expected reprocess warning toast when embedding dimension changes")
	}
}

func TestConfigSave_EmbeddingDimensionSame_NoWarning(t *testing.T) {
	h, _ := setupTestAdminHandler(t)
	h.config.Embedding.Dimension = 1024

	form := url.Values{
		"backend":    {""},
		"model":      {""},
		"dimension":  {"1024"}, // unchanged
		"max_tokens": {"512"},
		"batch_size": {"32"},
	}
	req := postForm("/admin/config/embedding", form)
	req = withChiParam(req, "section", "embedding")
	w := httptest.NewRecorder()

	h.ConfigSave(w, req)

	body := w.Body.String()
	if strings.Contains(body, "existing documents should be reprocessed") {
		t.Error("should not warn when embedding dimension is unchanged")
	}
}

func TestConfigSave_NonEmbeddingSection_NoWarning(t *testing.T) {
	h, _ := setupTestAdminHandler(t)
	h.config.Embedding.Dimension = 512

	form := url.Values{
		"top_k":              {"10"},
		"max_context_tokens": {"4096"},
	}
	req := postForm("/admin/config/rag", form)
	req = withChiParam(req, "section", "rag")
	w := httptest.NewRecorder()

	h.ConfigSave(w, req)

	body := w.Body.String()
	if strings.Contains(body, "existing documents should be reprocessed") {
		t.Error("should not warn about dimension when saving non-embedding section")
	}
}

// --- Backends ---

func TestBackendCreate(t *testing.T) {
	h, _ := setupTestAdminHandler(t)

	form := url.Values{
		"name":         {"local"},
		"base_url":     {"http://localhost:8081"},
		"api_key":      {"sk-test"},
		"enabled":      {"true"},
		"is_llama_cpp": {"true"},
	}
	req := postForm("/admin/backends", form)
	w := httptest.NewRecorder()

	h.BackendCreate(w, req)

	if len(h.config.Backends) != 1 {
		t.Fatalf("len(Backends) = %d, want 1", len(h.config.Backends))
	}
	b := h.config.Backends[0]
	if b.Name != "local" {
		t.Errorf("Name = %q, want %q", b.Name, "local")
	}
	if b.BaseURL != "http://localhost:8081" {
		t.Errorf("BaseURL = %q, want %q", b.BaseURL, "http://localhost:8081")
	}
	if !b.Enabled {
		t.Error("expected Enabled = true")
	}
	if !b.IsLlamaCpp {
		t.Error("expected IsLlamaCpp = true")
	}
}

func TestBackendUpdate(t *testing.T) {
	h, _ := setupTestAdminHandler(t)
	h.config.Backends = []config.BackendConfig{
		{Name: "local", BaseURL: "http://old", APIKey: "old-key", Enabled: true},
	}

	form := url.Values{
		"name":     {"local-updated"},
		"base_url": {"http://new"},
		"api_key":  {"new-key"},
		"enabled":  {"true"},
	}
	req := putForm("/admin/backends/local", form)
	req = withChiParam(req, "name", "local")
	w := httptest.NewRecorder()

	h.BackendUpdate(w, req)

	if len(h.config.Backends) != 1 {
		t.Fatalf("len(Backends) = %d, want 1", len(h.config.Backends))
	}
	b := h.config.Backends[0]
	if b.Name != "local-updated" {
		t.Errorf("Name = %q, want %q", b.Name, "local-updated")
	}
	if b.BaseURL != "http://new" {
		t.Errorf("BaseURL = %q, want %q", b.BaseURL, "http://new")
	}
	if b.APIKey != "new-key" {
		t.Errorf("APIKey = %q, want %q", b.APIKey, "new-key")
	}
}

func TestBackendUpdate_PreservesAPIKey(t *testing.T) {
	h, _ := setupTestAdminHandler(t)
	h.config.Backends = []config.BackendConfig{
		{Name: "local", BaseURL: "http://url", APIKey: "secret-key", Enabled: true},
	}

	form := url.Values{
		"name":     {"local"},
		"base_url": {"http://url"},
		"api_key":  {""}, // empty → preserve existing
		"enabled":  {"true"},
	}
	req := putForm("/admin/backends/local", form)
	req = withChiParam(req, "name", "local")
	w := httptest.NewRecorder()

	h.BackendUpdate(w, req)

	if h.config.Backends[0].APIKey != "secret-key" {
		t.Errorf("APIKey = %q, want preserved %q", h.config.Backends[0].APIKey, "secret-key")
	}
}

func TestBackendDelete(t *testing.T) {
	h, _ := setupTestAdminHandler(t)
	h.config.Backends = []config.BackendConfig{
		{Name: "keep", BaseURL: "http://a"},
		{Name: "remove", BaseURL: "http://b"},
	}

	req := httptest.NewRequest(http.MethodDelete, "/admin/backends/remove", nil)
	req = withChiParam(req, "name", "remove")
	w := httptest.NewRecorder()

	h.BackendDelete(w, req)

	if len(h.config.Backends) != 1 {
		t.Fatalf("len(Backends) = %d, want 1", len(h.config.Backends))
	}
	if h.config.Backends[0].Name != "keep" {
		t.Errorf("remaining backend = %q, want %q", h.config.Backends[0].Name, "keep")
	}
}

func TestBackendGetRow_NotFound(t *testing.T) {
	h, _ := setupTestAdminHandler(t)

	req := httptest.NewRequest(http.MethodGet, "/admin/backends/ghost", nil)
	req = withChiParam(req, "name", "ghost")
	w := httptest.NewRecorder()

	h.BackendGetRow(w, req)

	if w.Code != http.StatusNotFound {
		t.Errorf("status = %d, want 404", w.Code)
	}
}

// --- Documents ---

func TestDocumentList(t *testing.T) {
	h, s := setupTestAdminHandler(t)
	ctx := context.Background()

	s.CreateDocument(ctx, &domain.Document{
		ID: "doc-1", Title: "Test", Content: "content",
		Language: domain.LanguageES, Category: domain.CategoryGeneral,
	})

	req := httptest.NewRequest(http.MethodGet, "/admin/documents", nil)
	w := httptest.NewRecorder()

	h.DocumentList(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}
}

func TestDocumentDetail_NotFound(t *testing.T) {
	h, _ := setupTestAdminHandler(t)

	req := httptest.NewRequest(http.MethodGet, "/admin/documents/nonexistent", nil)
	req = withChiParam(req, "id", "nonexistent")
	w := httptest.NewRecorder()

	h.DocumentDetail(w, req)

	if w.Code != http.StatusNotFound {
		t.Errorf("status = %d, want 404", w.Code)
	}
}

func TestDocumentDelete(t *testing.T) {
	h, s := setupTestAdminHandler(t)
	ctx := context.Background()

	s.CreateDocument(ctx, &domain.Document{
		ID: "doc-del", Title: "Del", Content: "content",
		Language: domain.LanguageES, Category: domain.CategoryGeneral,
	})

	req := httptest.NewRequest(http.MethodDelete, "/admin/documents/doc-del", nil)
	req = withChiParam(req, "id", "doc-del")
	w := httptest.NewRecorder()

	h.DocumentDelete(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}

	// Verify it's gone.
	doc, _ := s.GetDocument(ctx, "doc-del")
	if doc != nil {
		t.Error("document should be deleted")
	}
}

func TestDocumentDelete_EmptyID(t *testing.T) {
	h, _ := setupTestAdminHandler(t)

	req := httptest.NewRequest(http.MethodDelete, "/admin/documents/", nil)
	req = withChiParam(req, "id", "")
	w := httptest.NewRecorder()

	h.DocumentDelete(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400", w.Code)
	}
}

// --- Chat ---

func TestChatSend_MissingParams(t *testing.T) {
	h, _ := setupTestAdminHandler(t)

	tests := []struct {
		name   string
		values url.Values
	}{
		{"missing message", url.Values{"session_id": {"s1"}}},
		{"missing session_id", url.Values{"message": {"hello"}}},
		{"both empty", url.Values{"message": {""}, "session_id": {""}}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := postForm("/admin/chat/send", tt.values)
			w := httptest.NewRecorder()

			h.ChatSend(w, req)

			if w.Code != http.StatusBadRequest {
				t.Errorf("status = %d, want 400", w.Code)
			}
		})
	}
}

func TestChatSend_NoAgent(t *testing.T) {
	h, _ := setupTestAdminHandler(t)

	sessionID := h.sessionStore.Create()

	form := url.Values{
		"message":    {"hello"},
		"session_id": {sessionID},
	}
	req := postForm("/admin/chat/send", form)
	w := httptest.NewRecorder()

	h.ChatSend(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}

	// ChatSend renders the user message HTML + SSE div for streaming.
	// Verify the response contains the user's message text.
	body := w.Body.String()
	if !strings.Contains(body, "hello") {
		t.Errorf("response should contain user message, got %q", body)
	}
	// Verify it references the session ID for SSE streaming.
	if !strings.Contains(body, sessionID) {
		t.Errorf("response should contain session ID for SSE, got %q", body)
	}
}

func TestChatStream_NotFound(t *testing.T) {
	h, _ := setupTestAdminHandler(t)

	req := httptest.NewRequest(http.MethodGet, "/admin/chat/stream/unknown", nil)
	req = withChiParam(req, "sessionID", "unknown")
	w := httptest.NewRecorder()

	h.ChatStream(w, req)

	if w.Code != http.StatusNotFound {
		t.Errorf("status = %d, want 404", w.Code)
	}
}

// TestChatStream_AfterFastSend verifies that the SSE stream survives in the map
// after ChatSend's goroutine finishes quickly (no agent configured). This was a
// race: the goroutine deleted the stream entry before HTMX could connect.
func TestChatStream_AfterFastSend(t *testing.T) {
	h, _ := setupTestAdminHandler(t)
	// h.agent is nil — the goroutine finishes immediately.

	sessionID := h.sessionStore.Create()

	// 1. ChatSend — registers the stream and spawns a goroutine that finishes fast.
	form := url.Values{"message": {"hi"}, "session_id": {sessionID}}
	sendReq := postForm("/admin/chat/send", form)
	sendW := httptest.NewRecorder()
	h.ChatSend(sendW, sendReq)

	if sendW.Code != http.StatusOK {
		t.Fatalf("ChatSend status = %d, want 200", sendW.Code)
	}

	// Small sleep to let the goroutine finish — simulates the delay before
	// HTMX opens the SSE connection.
	time.Sleep(50 * time.Millisecond)

	// 2. ChatStream — must still find the stream despite the goroutine having finished.
	streamReq := httptest.NewRequest(http.MethodGet, "/admin/chat/stream/"+sessionID, nil)
	streamReq = withChiParam(streamReq, "sessionID", sessionID)
	streamW := httptest.NewRecorder()
	h.ChatStream(streamW, streamReq)

	if streamW.Code != http.StatusOK {
		t.Fatalf("ChatStream status = %d, want 200 (stream should still exist)", streamW.Code)
	}

	body := streamW.Body.String()
	if !strings.Contains(body, "event: token") {
		t.Errorf("expected SSE token event, got %q", body)
	}
	if !strings.Contains(body, "event: done") {
		t.Errorf("expected SSE done event, got %q", body)
	}
	if !strings.Contains(body, "Agent not configured") {
		t.Errorf("expected placeholder message, got %q", body[:min(len(body), 500)])
	}
}

func TestChatReset(t *testing.T) {
	h, _ := setupTestAdminHandler(t)

	before := h.sessionStore.Count()

	req := postForm("/admin/chat/reset", url.Values{})
	w := httptest.NewRecorder()

	h.ChatReset(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}
	if h.sessionStore.Count() != before+1 {
		t.Errorf("session count = %d, want %d", h.sessionStore.Count(), before+1)
	}
}

// --- Crawler guard ---

// --- writeSSE ---

func TestWriteSSE_SingleLine(t *testing.T) {
	var buf bytes.Buffer
	writeSSE(&buf, "token", "hello")

	want := "event: token\ndata: hello\n\n"
	if got := buf.String(); got != want {
		t.Errorf("writeSSE =\n%q\nwant\n%q", got, want)
	}
}

func TestWriteSSE_MultiLine(t *testing.T) {
	var buf bytes.Buffer
	writeSSE(&buf, "tool_start", "<div>\n<span>test</span>\n</div>")

	want := "event: tool_start\ndata: <div>\ndata: <span>test</span>\ndata: </div>\n\n"
	if got := buf.String(); got != want {
		t.Errorf("writeSSE =\n%q\nwant\n%q", got, want)
	}
}

// --- ChatStream with tool events ---

func TestChatStream_ToolEvents(t *testing.T) {
	h, _ := setupTestAdminHandler(t)
	sessionID := "test-session"

	eventChan := make(chan agent.ToolEvent, 16)
	tokenChan := make(chan string, 64)
	stream := &chatStream{
		tokens: tokenChan,
		events: eventChan,
		cancel: func() {},
	}

	h.chatStreams.register(sessionID, stream)

	// Send tool events + token + close channels (simulates agent goroutine)
	go func() {
		eventChan <- agent.ToolEvent{Type: "start", CallID: "call-1", Name: "semantic_search", Args: `{"query":"test"}`}
		eventChan <- agent.ToolEvent{Type: "done", CallID: "call-1", Name: "semantic_search", Result: "found 3 results", Duration: 150 * time.Millisecond}
		close(eventChan)
		tokenChan <- "Here are the results"
		close(tokenChan)
	}()

	req := httptest.NewRequest(http.MethodGet, "/admin/chat/stream/"+sessionID, nil)
	req = withChiParam(req, "sessionID", sessionID)
	w := httptest.NewRecorder()

	h.ChatStream(w, req)

	body := w.Body.String()

	if !strings.Contains(body, "event: tool_start") {
		t.Error("expected tool_start SSE event")
	}
	if !strings.Contains(body, "event: tool_done") {
		t.Error("expected tool_done SSE event")
	}
	if !strings.Contains(body, "semantic_search") {
		t.Error("expected tool name in SSE data")
	}
	if !strings.Contains(body, "event: token") {
		t.Error("expected token SSE event")
	}
	if !strings.Contains(body, "event: done") {
		t.Error("expected done SSE event")
	}
}

func TestChatStream_NoToolEvents(t *testing.T) {
	h, _ := setupTestAdminHandler(t)
	sessionID := "test-no-tools"

	eventChan := make(chan agent.ToolEvent, 16)
	tokenChan := make(chan string, 64)
	stream := &chatStream{
		tokens: tokenChan,
		events: eventChan,
		cancel: func() {},
	}

	h.chatStreams.register(sessionID, stream)

	// No tool events, just token
	go func() {
		close(eventChan)
		tokenChan <- "Direct response"
		close(tokenChan)
	}()

	req := httptest.NewRequest(http.MethodGet, "/admin/chat/stream/"+sessionID, nil)
	req = withChiParam(req, "sessionID", sessionID)
	w := httptest.NewRecorder()

	h.ChatStream(w, req)

	body := w.Body.String()

	if strings.Contains(body, "event: tool_start") {
		t.Error("should not contain tool_start when no tools used")
	}
	if !strings.Contains(body, "event: token") {
		t.Error("expected token SSE event")
	}
	if !strings.Contains(body, "event: done") {
		t.Error("expected done SSE event")
	}
}

func TestCrawlerGuard_NilCrawler(t *testing.T) {
	h, _ := setupTestAdminHandler(t)
	// h.crawler is nil by default.

	req := httptest.NewRequest(http.MethodPost, "/admin/crawler/run", nil)
	w := httptest.NewRecorder()

	blocked := h.crawlerGuard(w, req, "start crawl")
	if !blocked {
		t.Error("crawlerGuard should return true when crawler is nil")
	}
	if w.Code != http.StatusServiceUnavailable {
		t.Errorf("status = %d, want 503", w.Code)
	}
}
