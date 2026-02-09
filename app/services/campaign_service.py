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
