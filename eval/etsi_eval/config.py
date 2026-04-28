"""Read the etsi-rag TOML config and surface backend settings.

The [eval] section is required and selects the judge backend. Using the same
backend/model for the agent and the judge is allowed but discouraged: it is
the canonical setup for self-preference bias. The eval tool only enforces
that [eval] is present and well-formed; the choice itself is the curator's.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BackendConfig:
    name: str
    base_url: str
    api_key: str


@dataclass(frozen=True)
class Runtime:
    """Everything the eval tool needs pulled out of config.toml."""

    agent_backend: BackendConfig
    agent_model: str
    judge_backend: BackendConfig
    judge_model: str
    db_path: Path
    server_base_url: str  # where etsi-rag serves /v1/chat/completions


def _resolve_backend(cfg: dict, name: str) -> BackendConfig:
    for b in cfg.get("backends", []):
        if b.get("name") == name and b.get("enabled", True):
            return BackendConfig(
                name=b["name"],
                base_url=b["base_url"].rstrip("/"),
                api_key=b.get("api_key", ""),
            )
    raise ValueError(
        f"no enabled backend named {name!r} in config.toml — check [[backends]] entries"
    )


def _require_eval(cfg: dict) -> tuple[str, str]:
    eval_cfg = cfg.get("eval")
    if eval_cfg is None:
        raise ValueError(
            "missing [eval] section in config.toml — add:\n"
            "  [eval]\n"
            '  judge_backend = "<name from [[backends]]>"\n'
            '  judge_model = "<model id>"'
        )
    backend = eval_cfg.get("judge_backend")
    model = eval_cfg.get("judge_model")
    missing = [k for k, v in (("judge_backend", backend), ("judge_model", model)) if not v]
    if missing:
        raise ValueError(
            f"[eval] section is missing required key(s): {', '.join(missing)}"
        )
    return backend, model


def load(project_root: Path) -> Runtime:
    """project_root = the etsi-rag/ directory (one that contains config/)."""
    cfg_path = project_root / "config" / "config.toml"
    if not cfg_path.exists():
        cfg_path = project_root / "config" / "config.default.toml"
    with cfg_path.open("rb") as fh:
        cfg = tomllib.load(fh)

    agent = cfg.get("agent", {})
    server = cfg.get("server", {})

    agent_backend = _resolve_backend(cfg, agent["backend"])

    judge_backend_name, judge_model = _require_eval(cfg)
    judge_backend = _resolve_backend(cfg, judge_backend_name)

    db_rel = Path(cfg.get("database", {}).get("path", "data/etsi-rag.db"))
    db_path = db_rel if db_rel.is_absolute() else (project_root / db_rel)

    host = server.get("host", "127.0.0.1")
    port = server.get("port", 3000)

    return Runtime(
        agent_backend=agent_backend,
        agent_model=agent["model"],
        judge_backend=judge_backend,
        judge_model=judge_model,
        db_path=db_path,
        server_base_url=f"http://{host}:{port}",
    )
