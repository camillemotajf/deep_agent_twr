from langchain.tools import tool
from app.config.container import request_service
from app.config.container import campaign_service


@tool
async def query_mongo_requests(
    hash: str | None = None,
    hashes: list[str] | None = None,
    limit: int = 1000
) -> list[dict]:
    """
    Retrieves recent MongoDB HTTP requests.

    Rules:
    - Use 'hash' for a single campaign
    - Use 'hashes' for multiple campaigns
    - Returns only requests with decision bots or unsafe
    """

    final_hashes = hashes or ([hash] if hash else [])

    cursor = await request_service.fetch_recent_flagged_requests(
        hashes=final_hashes,
        limit=limit
    )

    results = await cursor.to_list(length=limit)

    return results

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