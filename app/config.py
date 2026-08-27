from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    openai_api_key: str
    openai_base_url: str | None = None
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_dir: str = "./data/chroma"
    document_dir: str = "./data/documents"
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings()
