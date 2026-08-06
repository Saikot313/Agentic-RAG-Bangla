from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    llm_provider: str = "groq"  # "groq" | "local"
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    local_model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    hf_token: str = ""

    # Embeddings
    embedding_model: str = "intfloat/multilingual-e5-base"

    # Vector store
    faiss_index_dir: str = "storage/faiss_index"
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 5

    # App
    app_env: str = "development"


settings = Settings()
