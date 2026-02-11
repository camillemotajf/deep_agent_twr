from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, Boolean, DateTime, select, Integer
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime

class Base(DeclarativeBase):
    pass

class Campaign(Base):
    __tablename__ = "campaigns"

    campaign_pk: Mapped[int] = mapped_column(Integer, primary_key=True)
    hash: Mapped[str] = mapped_column(String, index=True)
    traffic_sources_fk: Mapped[int] = mapped_column(Integer, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, index=True)
    active: Mapped[bool] = mapped_column(Boolean, index=True)

class TrafficSource(Base):
    __tablename__ = "traffic_sources"

    traffic_sources_pk: Mapped[int] = mapped_column(Integer, primary_key=True)
    traffic_source: Mapped[str] = mapped_column(String, index=True)


class CampaignRepository:
    def __init__(self, session_factory):
        self.session_factory = session_factory

    async def get_recent_active_campaign_hashes(
        self,
        limit: int = 10,
        traffic_source: str | None = None
    ) -> list[str]:
        
        async with self.session_factory() as session:  # AsyncSession
            
            stmt = (
                select(Campaign.hash)
                .where(Campaign.active.is_(True))
                .order_by(Campaign.created_at.desc())
                .limit(limit)
            )

            if traffic_source:
                ## fazer a query na tabela de traffic sources
                tf_query = (
                    select(TrafficSource.traffic_sources_pk).where(TrafficSource.traffic_source == traffic_source)
                )
                tf_fk_result = await session.execute(tf_query)
                tf_fk = tf_fk_result.scalar_one_or_none() 

                stmt = stmt.where(Campaign.traffic_sources_fk == tf_fk)

            result = await session.execute(stmt)
            return result.scalars().all()
        
    async def get_traffic_source_by_hash(
            self,
            hash: str
    ) -> str:
        
        async with self.session_factory() as session:

            cpg_query = (
                select(Campaign.traffic_sources_fk)
                .where(Campaign.hash == hash)
            )

            cpg_result = await session.execute(cpg_query)
            cpg_tf_fk = cpg_result.scalar_one_or_none()

            tf_query = (
                    select(TrafficSource.traffic_source).where(TrafficSource.traffic_sources_pk == cpg_tf_fk)
                )
            tf_result = await session.execute(tf_query)
            tf = tf_result.scalar_one_or_none()

            return tf

