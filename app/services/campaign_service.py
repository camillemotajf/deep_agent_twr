from typing import List

from app.repositories.sql_repository import CampaignRepository

class CampaignService:
    def __init__(self, repository: CampaignRepository):
        self.repository = repository

    async def fetch_recent_active_campaigns(
        self,
        traffic_source: str | None = None,
        limit: int = 10
    ) -> list[str]:
        return await self.repository.get_recent_active_campaign_hashes(
            traffic_source=traffic_source,
            limit=limit
        )
    
    async def fetch_traffic_source_by_hash(
            self,
            hash: str
    ) -> str:
        return await self.repository.get_traffic_source_by_hash(
            hash=hash
        )
    async def fetch_recent_active_campaign_hashes_excluding(
            self,
            traffic_source: str,
            excluded_hashes: List[str],
            limit: int = 10
    ) -> List[str]:
        return await self.repository.get_recent_active_campaign_hashes_excluding(
            excluded_hashes=excluded_hashes,
            limit=limit,
            traffic_source=traffic_source
        )
