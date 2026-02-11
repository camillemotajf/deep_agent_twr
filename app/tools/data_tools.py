from typing import Dict, List
import uuid
import json
import os
from langchain.tools import tool
import pandas as pd
from app.config.container import request_service
from app.config.container import campaign_service
from app.tools.context_store import AnalysisContext

DATA_DIR = os.path.join(os.getcwd(), "files", "data")

@tool
def list_avaiable_datasets() -> List[str]:
    """
    Scans the data directory and returns a list of available log files.
    Use this when the user asks what data is available to analyze.
    """
    
    if not os.path.exists(DATA_DIR):
        return [f"Error: Directory {DATA_DIR}"]
    
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.json')]
    return files if files else ["No .json files found in data directory"]

@tool
def inspect_file_schema(filename: str) -> Dict[str, str]:
    """
    Use this to verify if a file contains the necessary fields (like 'user_agent', 'url')
    before committing to load it.
    """
    path = os.path.join(DATA_DIR, filename)
    
    if not os.path.exists(path):
        return {"error": "File not found"}
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            first_char = f.read(1)
            f.seek(0)
            
            sample = {}
            if first_char == '[':
                data = json.load(f)
                if isinstance(data, list) and len(data) > 0:
                    sample = data[0]
            else:
                line = f.readline()
                sample = json.loads(line)
                
        keys = list(sample.keys())
        return {
            "filename": filename,
            "status": "valid_structure",
            "columns_found": keys,
            "total_columns": len(keys),
            "sample_id": str(sample.get("id", "N/A"))
        }
    except Exception as e:
        return {"status": "invalid_format", "error": str(e)}
    

@tool
def load_dataset_into_context(filename: str, traffic_source: str) -> str:
    """
    Loads a specific dataset into the global memory (Context).
    
    REQUIRED: This tool MUST be called before any ML analysis can start.
    
    :param filename: The exact name of the file (e.g., 'google_logs.json')
    :param traffic_source: The source domain (e.g., 'google', 'facebook', 'organic'). 
                           This is crucial because it determines which ML model weights will be used later.
    """
    path = os.path.join(DATA_DIR, filename)
    
    if not os.path.exists(path):
        return f"Error: File {filename} does not exist in {DATA_DIR}."
        
    try:
        df = pd.read_json(path)
        
        required_cols = ["id", "user_agent"] 
        missing = [c for c in required_cols if c not in df.columns]
        
        if missing:
            return f"Error: The file is missing required columns for analysis: {missing}"
            
        AnalysisContext.load_raw_data(df, traffic_source, filename)
        
        return (
            f"SUCCESS: Loaded '{filename}' into memory.\n"
            f" - Rows: {len(df)}\n"
            f" - Traffic Source: '{traffic_source}'\n"
            "State updated. You can now transfer control to the 'Metrics Analyst' agent."
        )
        
    except ValueError as e:
        return f"Error parsing JSON content: {str(e)}"
    except Exception as e:
        return f"System Critical Error: {str(e)}"
    
@tool
def check_context_status() -> str:
    """
    Checks what data is currently loaded in memory.
    Use this to avoid reloading the same file if the user asks a follow-up question.
    """
    return AnalysisContext.get_status()

@tool
async def traffic_source_by_campaign(
    hash: str
) -> str:
    """
    Gets the traffic source name for the campaign from its hash
    Use this when a hash is provided by user and it's necessary searching for the specific ML model.
    Then, use the 'query_mongo_requests' tool to fetch the request to be analised by the ML model
    
    :param hash: campaign hash
    :type hash: str
    :return: the name of the traffic source
    :rtype: str
    """

    try:
        result = await campaign_service.fetch_traffic_source_by_hash(hash)
    except ValueError as e:
        return f"Error getting the traffic source name for {hash} campaign"
    
    AnalysisContext.set_traffic_source(result)

    return f"Success on setting traffic source in context. Traffic Source: {result}"


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
    
    Rules:
    - Use the 'traffic source' for context
    - Use 'hash' for a single campaign
    - Use 'hashes' for multiple campaigns
    - Returns only requests with decision bots or unsafe
    """

    final_hashes = hashes or ([hash] if hash else [])

    if not final_hashes:
        return "Error: You must provide at least one 'hash' or a list of 'hashes'."

    try:
        cursor = await request_service.fetch_recent_flagged_requests(
            hashes=final_hashes,
            limit=limit
        )
        results = await cursor.to_list(length=limit)

        if not results:
            return f"No data found in MongoDB for hashes: {final_hashes}"

        df = pd.DataFrame(results)

        AnalysisContext.set_mongo_data(
            df=df, 
            source=traffic_source, 
        )

        return (
            f"SUCCESS: Loaded {len(df)} requests into AnalysisContext.\n"
            f"Sources: {traffic_source} | Hashes: {len(final_hashes)}\n"
            "Action Required: Delegate to 'Metrics Analyst' agent to run ML inference now."
        )

    except Exception as e:
        return f"Error loading data from Mongo: {str(e)}"

@tool
async def query_sql_campaigns(
    traffic_source: str | None = None,
    limit: int = 10
) -> list[str]:
    """    
    Use this tool when the user asks to analyze traffic but doesn't provide a specific hash.
    It returns a list of available campaign hashes and their metadata.
    
    Returns: A list of campaigns hashes to use query_mongo_requests to search for their data.
    """
    try:
        campaigns = await campaign_service.fetch_recent_active_campaigns(
            traffic_source=traffic_source,
            limit=limit
        )
        
        if not campaigns:
            return f"No active campaigns found for source: {traffic_source}."

        return campaigns

    except Exception as e:
        return [f"Error querying SQL: {str(e)}"]
