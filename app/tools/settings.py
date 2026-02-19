
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    MODELS_PATH: str
    TRANSFORMER_MODEL: str

    class Config:
        env_file = ".env"  
        extra = "ignore" 

settings = Settings()  