from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    ALLOWED_ORIGINS: str = "*"

    POSTGRES_HOST: str = "rawg-db.c5gics8qchki.eu-north-1.rds.amazonaws.com "
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "rawg"
    POSTGRES_USER: str = "joseph"
    POSTGRES_PASSWORD: str = "Joseph123!"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

settings = Settings()
