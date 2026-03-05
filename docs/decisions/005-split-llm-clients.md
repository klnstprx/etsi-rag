# ADR-005: Split LLM Clients

## Status
Accepted

## Context
Embedding and chat may use different backends (e.g., local embedding model + remote chat model). A single LLM client per stage is too rigid.

## Decision
Split LLM dependencies at consumption points:
- RAG pipeline takes `embedder` + `chatter` (both `llm.LLM` interface)
- API handler takes `embeddingClient` + `chatClient`
- `main.go` resolves backends by name via `clientForStage()` and wires them

Backend config (`[[backends]]`) defines named endpoints. Stage config (`[embedding]`, `[chat]`, `[agent]`) references a backend by name and adds model-specific settings.

## Consequences
- Mix and match backends freely (local embed + remote chat)
- Each stage can have its own model, temperature, max_tokens
- `llm.LLM` interface stays small (Embed, Chat, ChatStream)
- `main.go` is the only place that knows about backend resolution
