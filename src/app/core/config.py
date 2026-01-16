from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import PostgresDsn, computed_field
from typing import Optional

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True)

    # API
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Agentic RAG Platform"
    
    # POSTGRES
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "rag_user"
    POSTGRES_PASSWORD: str = "rag_password"
    POSTGRES_DB: str = "rag_db"

    @computed_field
    def SQLALCHEMY_DATABASE_URI(self) -> PostgresDsn:
        return PostgresDsn.build(
            scheme="postgresql+asyncpg",
            username=self.POSTGRES_USER,
            password=self.POSTGRES_PASSWORD,
            host=self.POSTGRES_SERVER,
            port=self.POSTGRES_PORT,
            path=self.POSTGRES_DB,
        )

    # QDRANT
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: Optional[str] = None
    
    # LLM (Defaults to Ollama/Local)
    LLM_MODEL: str = "ollama/llama3"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    OLLAMA_BASE_URL: str = "http://localhost:11434"

settings = Settings()
