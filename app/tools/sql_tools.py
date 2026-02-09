from langchain.tools import tool
from app.config.container import campaign_service

@tool
async def query_sql_campaigns(
    traffic_source: str | None = None,
    limit: int = 10
) -> list[str]:
    """
    Retrieves recent active campaign hashes.
    """
    return await campaign_service.fetch_recent_active_campaigns(
        traffic_source=traffic_source,
        limit=limit
    )
