import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    llm_model: str
    openai_api_key: str | None
    ollama_base_url: str
    data_dir: Path
    db_path: Path
    chroma_dir: Path
    cache_dir: Path
    upload_dir: Path
    embedding_model: str
    extract_concurrency: int
    retrieve_top_k: int


def get_settings() -> Settings:
    provider = os.getenv("LLM_PROVIDER", "")
    data_dir = Path(os.getenv("VERITAS_DATA_DIR", "data"))
    for sub in ("", "chroma", "cache", "uploads"):
        (data_dir / sub).mkdir(parents=True, exist_ok=True)
    return Settings(
        llm_provider=provider,
        llm_model=os.getenv("LLM_MODEL", ""),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        data_dir=data_dir,
        db_path=data_dir / "veritas.db",
        chroma_dir=data_dir / "chroma",
        cache_dir=data_dir / "cache",
        upload_dir=data_dir / "uploads",
        embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        extract_concurrency=int(os.getenv("EXTRACT_CONCURRENCY", "4")),
        retrieve_top_k=int(os.getenv("RETRIEVE_TOP_K", "8")),
    )
