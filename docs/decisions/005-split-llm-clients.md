# ADR-005: Split LLM Clients

## Status
Accepted

## Context
Embedding and chat may use different backends (e.g., local embedding model + remote chat model). A single LLM client per stage is too rigid. Additionally, a monolithic `llm.Client` carries unused fields for each role: a chat-only client has an unused `embedBatchSize`; an embed-only client has unused `maxTokens`/`temperature`.

## Decision
Split the concrete implementations and expose an interface only where a consumer actually needs to swap impls:
- **Interface**: `llm.Embedder` (Embed, EmbedBatch) — the RAG pipeline depends on this so tests can substitute a stub. No `Chatter` interface is exported; the agent depends on `*llm.ChatClient` concretely.
- **Concrete types**: `llm.EmbedClient` (embedding-specific fields: batchSize) + `llm.ChatClient` (chat-specific fields: maxTokens, temperature). `ChatWithTools` and `ChatStreamWithTools` live on `ChatClient`.
- RAG pipeline takes `llm.Embedder` only (pure retrieval service, no chat client)
- API handler takes `llm.Embedder` + delegates chat to `agent.Agent`
- Agent takes `*llm.ChatClient` concretely (needs the tool-calling methods)
- The `server` package resolves backends via `resolveEmbedClient()` and `resolveChatClient()` (in `internal/server/wiring.go`); the reranker stage adds `resolveRerankClient()` for the llama.cpp `/v1/rerank` extension.

Backend config (`[[backends]]`) defines named endpoints. Stage config (`[embedding]`, `[agent]`) references a backend by name and adds model-specific settings. The `[chat]` stage was removed — the agent is the single entry point for all conversations.

## Consequences
- Mix and match backends freely (local embed + remote chat)
- Each stage can have its own model, temperature, max_tokens
- No unused fields on either client type
- The `server` package is the only place that knows about backend resolution
