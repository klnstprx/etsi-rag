package admin

import (
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
		AgentHandle:  agent.NewHandle(nil),
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

func TestConfigSave_Reranker(t *testing.T) {
	h, _ := setupTestAdminHandler(t)
	// Need an enabled llama-cpp backend for Validate to accept the reranker.
	h.config.Backends = []config.BackendConfig{
		{Name: "llama", BaseURL: "http://localhost:1337/v1", Enabled: true, IsLlamaCpp: true},
	}
	h.config.Retrieval.TopK = 10

	form := url.Values{
		"enabled":         {"true"},
		"backend":         {"llama"},
		"model":           {"qwen-reranker"},
		"top_n":           {"30"},
		"max_tokens":      {"4096"},
		"reserved_tokens": {"16"},
		"timeout_s":       {"30"},
	}
	req := postForm("/admin/config/reranker", form)
	req = withChiParam(req, "section", "reranker")
	w := httptest.NewRecorder()

	h.ConfigSave(w, req)

	if !h.config.Reranker.Enabled {
		t.Error("Reranker.Enabled = false, want true")
	}
	if h.config.Reranker.Backend != "llama" {
		t.Errorf("Reranker.Backend = %q, want llama", h.config.Reranker.Backend)
	}
	if h.config.Reranker.Model != "qwen-reranker" {
		t.Errorf("Reranker.Model = %q, want qwen-reranker", h.config.Reranker.Model)
	}
	if h.config.Reranker.TopN != 30 {
		t.Errorf("Reranker.TopN = %d, want 30", h.config.Reranker.TopN)
	}
	if h.config.Reranker.MaxTokens != 4096 {
		t.Errorf("Reranker.MaxTokens = %d, want 4096", h.config.Reranker.MaxTokens)
	}
	if h.config.Reranker.Timeout != 30*time.Second {
		t.Errorf("Reranker.Timeout = %s, want 30s", h.config.Reranker.Timeout)
	}
}

// TestConfigSave_Reranker_RejectsNonLlamaCppBackend exercises the
// validate-on-save path: enabling the reranker against an OpenAI backend must
// be rejected and the in-memory config must revert.
func TestConfigSave_Reranker_RejectsNonLlamaCppBackend(t *testing.T) {
	h, _ := setupTestAdminHandler(t)
	h.config.Backends = []config.BackendConfig{
		{Name: "openai", BaseURL: "https://api.openai.com/v1", Enabled: true, IsLlamaCpp: false},
	}
	h.config.Retrieval.TopK = 10
	originalEnabled := h.config.Reranker.Enabled

	form := url.Values{
		"enabled":         {"true"},
		"backend":         {"openai"},
		"model":           {"some-model"},
		"top_n":           {"30"},
		"reserved_tokens": {"16"},
		"timeout_s":       {"30"},
	}
	req := postForm("/admin/config/reranker", form)
	req = withChiParam(req, "section", "reranker")
	w := httptest.NewRecorder()

	h.ConfigSave(w, req)

	if h.config.Reranker.Enabled != originalEnabled {
		t.Errorf("Reranker.Enabled = %v, want %v (revert on validation failure)",
			h.config.Reranker.Enabled, originalEnabled)
	}
}

func TestConfigSave_Reranker_RejectsTopNUnderTopK(t *testing.T) {
	h, _ := setupTestAdminHandler(t)
	h.config.Backends = []config.BackendConfig{
		{Name: "llama", BaseURL: "http://localhost:1337/v1", Enabled: true, IsLlamaCpp: true},
	}
	h.config.Retrieval.TopK = 10
	originalEnabled := h.config.Reranker.Enabled

	form := url.Values{
		"enabled":         {"true"},
		"backend":         {"llama"},
		"model":           {"qwen-reranker"},
		"top_n":           {"5"},
		"reserved_tokens": {"16"},
		"timeout_s":       {"30"},
	}
	req := postForm("/admin/config/reranker", form)
	req = withChiParam(req, "section", "reranker")
	w := httptest.NewRecorder()

	h.ConfigSave(w, req)

	if h.config.Reranker.Enabled != originalEnabled {
		t.Errorf("Reranker.Enabled = %v, want %v (revert on validation failure)",
			h.config.Reranker.Enabled, originalEnabled)
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

func TestConfigSave_Agent_ReserveTokens(t *testing.T) {
	h, _ := setupTestAdminHandler(t)

	form := url.Values{
		"backend":                {""},
		"model":                  {""},
		"max_tokens":             {"1024"},
		"temperature":            {"0.5"},
		"context_window":         {"8192"},
		"max_iterations":         {"10"},
		"tool_output_max_tokens": {"0"},
		"history_reserve_tokens": {"900"},
		"user_reserve_tokens":    {"700"},
		"system_prompt":          {""},
	}
	req := postForm("/admin/config/agent", form)
	req = withChiParam(req, "section", "agent")
	w := httptest.NewRecorder()

	h.ConfigSave(w, req)

	if h.config.Agent.HistoryReserveTokens != 900 {
		t.Errorf("HistoryReserveTokens = %d, want 900", h.config.Agent.HistoryReserveTokens)
	}
	if h.config.Agent.UserReserveTokens != 700 {
		t.Errorf("UserReserveTokens = %d, want 700", h.config.Agent.UserReserveTokens)
	}
}

// fakeCapacities satisfies Capacities for rendering tests.
type fakeCapacities struct{ agentCtx, embMax, rerankMax int }

func (f fakeCapacities) AgentContextWindow() int { return f.agentCtx }
func (f fakeCapacities) EmbeddingMaxTokens() int { return f.embMax }
func (f fakeCapacities) RerankerMaxTokens() int  { return f.rerankMax }

// TestConfigSave_Agent_ContextWindowAutoRoundTrip locks the 0 = auto contract:
// a form that posts context_window=0 must keep 0 in config (the probe stays in
// charge), never the effective value the UI displays alongside it.
func TestConfigSave_Agent_ContextWindowAutoRoundTrip(t *testing.T) {
	h, _ := setupTestAdminHandler(t)
	h.config.Agent.ContextWindow = 0

	form := url.Values{
		"backend":                {""},
		"model":                  {""},
		"max_tokens":             {"1024"},
		"temperature":            {"0.5"},
		"context_window":         {"0"},
		"max_iterations":         {"10"},
		"tool_output_max_tokens": {"0"},
		"history_reserve_tokens": {"1280"},
		"user_reserve_tokens":    {"512"},
		"system_prompt":          {""},
	}
	req := postForm("/admin/config/agent", form)
	req = withChiParam(req, "section", "agent")
	w := httptest.NewRecorder()

	h.ConfigSave(w, req)

	if h.config.Agent.ContextWindow != 0 {
		t.Errorf("ContextWindow = %d, want 0 (auto must survive a save)", h.config.Agent.ContextWindow)
	}
}

// TestConfigSections_ContextWindowShowsIntentNotEffective locks the rendering
// contract: the editable field carries the config value (operator intent, 0 =
// auto) while the live probe-resolved value appears only in the hint.
func TestConfigSections_ContextWindowShowsIntentNotEffective(t *testing.T) {
	cfg := &config.Config{}
	cfg.Agent.ContextWindow = 0
	caps := fakeCapacities{agentCtx: 16384, embMax: 4096, rerankMax: 2048}

	sections := configToSections(cfg, caps)
	agent := findSection(sections, "agent")
	if agent == nil {
		t.Fatal("agent section not found")
	}
	for _, f := range agent.Fields {
		if f.Key != "context_window" {
			continue
		}
		if f.Value != "0" {
			t.Errorf("context_window field value = %q, want \"0\" (operator intent, not effective)", f.Value)
		}
		if !strings.Contains(f.Hint, "16384") {
			t.Errorf("context_window hint = %q, want it to carry the effective value 16384", f.Hint)
		}
		return
	}
	t.Fatal("context_window field not found")
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

// reprocessWarning is the toast fragment that signals stale stored embeddings.
// Tests assert on the prefix so wording tweaks don't churn every assertion.
const reprocessWarning = "Embedding model or dimension changed"

func TestConfigSave_EmbeddingDimensionChange_WarnsReprocess(t *testing.T) {
	h, _ := setupTestAdminHandler(t)
	h.config.Embedding.Dimension = 512 // initial
	h.config.Embedding.Model = "bge-small"

	form := url.Values{
		"backend":    {""},
		"model":      {"bge-small"},
		"dimension":  {"1024"}, // changed
		"max_tokens": {"512"},
		"batch_size": {"32"},
	}
	req := postForm("/admin/config/embedding", form)
	req = withChiParam(req, "section", "embedding")
	w := httptest.NewRecorder()

	h.ConfigSave(w, req)

	body := w.Body.String()
	if !strings.Contains(body, reprocessWarning) {
		t.Error("expected reprocess warning toast when embedding dimension changes")
	}
}

func TestConfigSave_EmbeddingModelChange_WarnsReprocess(t *testing.T) {
	h, _ := setupTestAdminHandler(t)
	h.config.Embedding.Dimension = 1024
	h.config.Embedding.Model = "bge-small"

	form := url.Values{
		"backend":    {""},
		"model":      {"bge-large"}, // changed, dim same
		"dimension":  {"1024"},
		"max_tokens": {"512"},
		"batch_size": {"32"},
	}
	req := postForm("/admin/config/embedding", form)
	req = withChiParam(req, "section", "embedding")
	w := httptest.NewRecorder()

	h.ConfigSave(w, req)

	body := w.Body.String()
	if !strings.Contains(body, reprocessWarning) {
		t.Error("expected reprocess warning toast when embedding model changes (vectors live in different semantic space)")
	}
}

func TestConfigSave_EmbeddingDimensionSame_NoWarning(t *testing.T) {
	h, _ := setupTestAdminHandler(t)
	h.config.Embedding.Dimension = 1024
	h.config.Embedding.Model = "bge-small"

	form := url.Values{
		"backend":    {""},
		"model":      {"bge-small"}, // unchanged
		"dimension":  {"1024"},      // unchanged
		"max_tokens": {"512"},
		"batch_size": {"32"},
	}
	req := postForm("/admin/config/embedding", form)
	req = withChiParam(req, "section", "embedding")
	w := httptest.NewRecorder()

	h.ConfigSave(w, req)

	body := w.Body.String()
	if strings.Contains(body, reprocessWarning) {
		t.Error("should not warn when embedding model and dimension are unchanged")
	}
}

func TestConfigSave_NonEmbeddingSection_NoWarning(t *testing.T) {
	h, _ := setupTestAdminHandler(t)
	h.config.Embedding.Dimension = 512
	h.config.Embedding.Model = "bge-small"

	form := url.Values{
		"top_k":              {"10"},
		"max_context_tokens": {"4096"},
	}
	req := postForm("/admin/config/rag", form)
	req = withChiParam(req, "section", "rag")
	w := httptest.NewRecorder()

	h.ConfigSave(w, req)

	body := w.Body.String()
	if strings.Contains(body, reprocessWarning) {
		t.Error("should not warn about embeddings when saving non-embedding section")
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
		Language: domain.LanguageES,
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
		Language: domain.LanguageES,
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
//
// The admin chat surface is now just a shell: the browser talks to the
// proprietary POST /v1/agent/chat endpoint directly (see static/chat.js and
// the api package tests). Server-side, only the page render + reset remain.

func TestChatPage(t *testing.T) {
	h, _ := setupTestAdminHandler(t)

	req := httptest.NewRequest(http.MethodGet, "/admin/chat", nil)
	w := httptest.NewRecorder()
	h.ChatPage(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", w.Code)
	}
	body := w.Body.String()
	for _, want := range []string{`id="chat-form"`, "data-session-id=", `id="chat-messages"`, "/admin/static/chat.js"} {
		if !strings.Contains(body, want) {
			t.Errorf("chat page missing %q", want)
		}
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
