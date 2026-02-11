import uuid
import json
import os
from langchain.tools import tool
from app.config.container import request_service
from app.config.container import campaign_service

@tool
async def traffic_source_by_campaign(
    hash: str
) -> str:
    """
    Gets the traffic source name for the campaign from its hash
    
    :param hash: campaign hash
    :type hash: str
    :return: the name of the traffic source
    :rtype: str
    """

    return await campaign_service.fetch_traffic_source_by_hash(hash)


@tool
async def query_mongo_requests(
    traffic_source: str,
    hash: str | None = None,
    hashes: list[str] | None = None,
    limit: int = 1000
) -> list[dict]:
    """
    Retrieves recent MongoDB HTTP requests and saves them to a temporary file.
    Returns the file path to be used by analysis tools.
    
    CRITICAL: You MUST provide 'hash' or 'hashes'. 
    If you only know the traffic_source, use query_sql_campaigns FIRST to get the hashes.


    Rules:
    - Use the 'traffic source' for context
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

    if not results:
        return {"status": "empty", "count": 0}
    
    os.makedirs("tool_outputs", exist_ok=True)
    filename = f"tool_outputs/mongo_requests_{uuid.uuid4().hex}.json"

    with open(filename, "w") as f:
        from bson import json_util 
        f.write(json_util.dumps(results))

    return {
        "status": "success",
        "file_path": filename,        
        "traffic_source": traffic_source,
        "count": len(results),
        "preview": results[:2]     
    }

@tool
async def query_sql_campaigns(
    traffic_source: str | None = None,
    limit: int = 10
) -> list[str]:
    """
    Retrieves recent Campaigns Hashes.

    Rules:
    - Use 'traffic_source' to return a list of the most recent campaigns created 
    - Use 'limit' for setting  the number of campaigns hashes to be given
    - Returns the campaigns. Use this campaigns to trigger their requests in mongo db
    """
    return await campaign_service.fetch_recent_active_campaigns(
        traffic_source=traffic_source,
        limit=limit
    )