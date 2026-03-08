from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MONGO_URI: str
    SQL_DATABASE_URL: str
    COL_REQUEST: str
    DB_NAME: str
    TRANSFORMER_MODEL: str
    REPOSITORY_PATH: str
    SHODAN_API_KEY: str
    API_BROWSER_KEY: str
    API_BROWSER_URL: str

    class Config:
        env_file = ".env"  
        extra = "ignore" 

settings = Settings()  