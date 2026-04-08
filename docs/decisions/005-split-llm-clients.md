# ADR-005: Split LLM Clients

## Status
Accepted

## Context
Embedding and chat may use different backends (e.g., local embedding model + remote chat model). A single LLM client per stage is too rigid. Additionally, a monolithic `llm.Client` carries unused fields for each role: a chat-only client has an unused `embedBatchSize`; an embed-only client has unused `maxTokens`/`temperature`.

## Decision
Split both the interfaces and the concrete implementations:
- **Interfaces**: `llm.Embedder` (Embed, EmbedBatch) + `llm.Chatter` (Chat, ChatStream)
- **Concrete types**: `llm.EmbedClient` (embedding-specific fields: batchSize) + `llm.ChatClient` (chat-specific fields: maxTokens, temperature). `ChatWithTools` lives on `ChatClient`.
- RAG pipeline takes `llm.Embedder` only (pure retrieval service, no chatter)
- API handler takes `llm.Embedder` + delegates chat to `agent.Agent`
- Agent takes `*llm.ChatClient` concretely (needs `ChatWithTools`)
- `main.go` resolves backends via `embedClientForStage()` and `chatClientForStage()`

Backend config (`[[backends]]`) defines named endpoints. Stage config (`[embedding]`, `[agent]`) references a backend by name and adds model-specific settings. The `[chat]` stage was removed — the agent is the single entry point for all conversations.

## Consequences
- Mix and match backends freely (local embed + remote chat)
- Each stage can have its own model, temperature, max_tokens
- No unused fields on either client type
- `main.go` is the only place that knows about backend resolution
