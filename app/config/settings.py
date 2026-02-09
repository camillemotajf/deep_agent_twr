from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MONGO_URI: str
    WORD2VEC_MODEL: str
    SQL_DATABASE_URL: str
    MODELS_BASE_PATH: str = "models_ia"
    COL_REQUEST: str
    DB_NAME: str

    class Config:
        env_file = ".env"  
        extra = "ignore" 

settings = Settings()  