"""Config loader tests — focus on the required [eval] section."""

from __future__ import annotations

from pathlib import Path

import pytest

from etsi_eval.config import load


def _write(tmp_path: Path, body: str) -> Path:
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    (cfg_dir / "config.toml").write_text(body, encoding="utf-8")
    return tmp_path


_BACKENDS = """
[server]
host = "127.0.0.1"
port = 3000

[[backends]]
name = "local-llama"
base_url = "http://localhost:1337/v1"
api_key = ""
enabled = true

[[backends]]
name = "remote-claude"
base_url = "https://api.anthropic.com/v1"
api_key = "sk-test"
enabled = true

[agent]
backend = "local-llama"
model = "agent-model"

[database]
path = "data/etsi-rag.db"
"""


def test_load_happy_path(tmp_path: Path):
    body = _BACKENDS + """
[eval]
judge_backend = "remote-claude"
judge_model = "claude-sonnet-4-6"
"""
    rt = load(_write(tmp_path, body))
    assert rt.agent_backend.name == "local-llama"
    assert rt.judge_backend.name == "remote-claude"
    assert rt.judge_backend.base_url == "https://api.anthropic.com/v1"
    assert rt.judge_model == "claude-sonnet-4-6"


def test_missing_eval_section_errors(tmp_path: Path):
    with pytest.raises(ValueError, match=r"\[eval\]"):
        load(_write(tmp_path, _BACKENDS))


def test_missing_judge_backend_key_errors(tmp_path: Path):
    body = _BACKENDS + """
[eval]
judge_model = "claude-sonnet-4-6"
"""
    with pytest.raises(ValueError, match="judge_backend"):
        load(_write(tmp_path, body))


def test_missing_judge_model_key_errors(tmp_path: Path):
    body = _BACKENDS + """
[eval]
judge_backend = "remote-claude"
"""
    with pytest.raises(ValueError, match="judge_model"):
        load(_write(tmp_path, body))


def test_unknown_judge_backend_errors(tmp_path: Path):
    body = _BACKENDS + """
[eval]
judge_backend = "does-not-exist"
judge_model = "claude-sonnet-4-6"
"""
    with pytest.raises(ValueError, match="does-not-exist"):
        load(_write(tmp_path, body))


def test_judge_can_be_same_as_agent(tmp_path: Path):
    """Self-judge is allowed but discouraged — config doesn't forbid it."""
    body = _BACKENDS + """
[eval]
judge_backend = "local-llama"
judge_model = "agent-model"
"""
    rt = load(_write(tmp_path, body))
    assert rt.judge_backend.name == rt.agent_backend.name
    assert rt.judge_model == rt.agent_model


def test_pricing_defaults_to_zero(tmp_path: Path):
    body = _BACKENDS + """
[eval]
judge_backend = "remote-claude"
judge_model = "claude-sonnet-4-6"
"""
    rt = load(_write(tmp_path, body))
    assert rt.judge_pricing.input_per_mtok == 0.0
    assert rt.judge_pricing.output_per_mtok == 0.0
    assert not rt.judge_pricing.enabled


def test_pricing_loaded_when_set(tmp_path: Path):
    body = _BACKENDS + """
[eval]
judge_backend = "remote-claude"
judge_model = "claude-sonnet-4-6"
judge_input_price_per_mtok = 3.0
judge_output_price_per_mtok = 15.0
"""
    rt = load(_write(tmp_path, body))
    assert rt.judge_pricing.input_per_mtok == 3.0
    assert rt.judge_pricing.output_per_mtok == 15.0
    assert rt.judge_pricing.enabled
