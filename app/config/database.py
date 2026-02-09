from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker
)
from app.config.settings import settings

engine = create_async_engine(
    settings.SQL_DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    expire_on_commit=False
)
