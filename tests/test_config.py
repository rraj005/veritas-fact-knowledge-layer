from veritas.config import get_settings


def test_defaults(monkeypatch, tmp_path):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("VERITAS_DATA_DIR", str(tmp_path))
    s = get_settings()
    assert s.llm_provider == ""
    assert s.retrieve_top_k == 8
    assert s.db_path == tmp_path / "veritas.db"


def test_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o")
    monkeypatch.setenv("VERITAS_DATA_DIR", str(tmp_path))
    s = get_settings()
    assert s.llm_provider == "openai"
    assert s.llm_model == "gpt-4o"


def test_ollama_base_url_default(monkeypatch, tmp_path):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.setenv("VERITAS_DATA_DIR", str(tmp_path))
    s = get_settings()
    assert s.ollama_base_url == "http://localhost:11434"


def test_ollama_base_url_override(monkeypatch, tmp_path):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://myhost:9999")
    monkeypatch.setenv("VERITAS_DATA_DIR", str(tmp_path))
    s = get_settings()
    assert s.ollama_base_url == "http://myhost:9999"
