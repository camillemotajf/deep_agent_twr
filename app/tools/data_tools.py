from typing import Dict, List
import uuid
import json
import os
from langchain.tools import tool
import pandas as pd
from app.config.container import request_service
from app.config.container import campaign_service
from app.tools.context_store import AnalysisContext
from app.tools.pydantic_models import (
    QueryMongoRequestsInput, 
    QueryMongoRequestsOutput,
    QuerySqlCampaignsInput,
    QuerySqlCampaignsOutput,
    TrafficSourceByCampaignInput,
    TrafficSourceByCampaignOutput,
    QueryRequestsForTrainingInput,
    QueryRequestsForTrainingOutput
)

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

@tool(args_schema=TrafficSourceByCampaignInput)
async def traffic_source_by_campaign(
    **kwargs
) -> TrafficSourceByCampaignOutput:
    """
    Gets the traffic source name for the campaign from its hash
    Use this when a hash is provided by user and it's necessary searching for the specific ML model.
    Then, use the 'query_mongo_requests' tool to fetch the request to be analised by the ML model
    
    :param hash: campaign hash
    :type hash: str
    :return: the name of the traffic source
    :rtype: str
    """
    input_model = TrafficSourceByCampaignInput(**kwargs)
    try:
        result = await campaign_service.fetch_traffic_source_by_hash(input_model.hash)
    except ValueError as e:
        return TrafficSourceByCampaignOutput(success=False, message=f"Error getting the traffic source name for {input_model.hash} campaign")
    
    AnalysisContext.set_traffic_source(result)

    return TrafficSourceByCampaignOutput(success=True, traffic_source=result, message=f"Success on setting traffic source in context. Traffic Source: {result}")

@tool(args_schema=QueryRequestsForTrainingInput)
async def query_requests_for_training(
    **kwargs
) -> QueryRequestsForTrainingOutput:
    """
    Docstring para query_requests_for_training
    
    :param traffic_source: Descrição
    :type traffic_source: str | None
    :param hashes: Descrição
    :type hashes: list[str] | None
    :param limit: Descrição
    :type limit: int
    :return: Descrição
    :rtype: list[dict]
    """
    input_model = QueryRequestsForTrainingInput(**kwargs)
    if not input_model.traffic_source and not input_model.hashes:
        return QueryRequestsForTrainingOutput(success=False, message="Error: You must provide at least one of the oprions: traffic_source or hashes")
    
    elif input_model.traffic_source:
        campaigns = await campaign_service.fetch_recent_active_campaigns(traffic_source=input_model.traffic_source, limit=50)
    elif input_model.hashes:
        campaigns = input_model.hashes
    
    results = await request_service.fetch_training_sample_by_hashes(campaigns)

    return QueryRequestsForTrainingOutput(success=True, results=results)


@tool(args_schema=QueryMongoRequestsInput)
async def query_mongo_requests(
    **kwargs
) -> QueryMongoRequestsOutput:
    """
    Retrieves recent MongoDB HTTP requests and saves them to a temporary file.
    Returns the file path to be used by analysis tools.
    
    Rules:
    - Use the 'traffic source' for context
    - Use 'hash' for a single campaign
    - Use 'hashes' for multiple campaigns
    - Returns only requests with decision bots or unsafe
    """
    
    input_model = QueryMongoRequestsInput(**kwargs)

    final_hashes = input_model.hashes or ([input_model.hash] if input_model.hash else [])

    if not final_hashes:
        return QueryMongoRequestsOutput(success=False, message="Error: You must provide at least one 'hash' or a list of 'hashes'.")

    try:
        cursor = await request_service.fetch_recent_flagged_requests(
            hashes=final_hashes,
            limit=input_model.limit
        )
        results = await cursor.to_list(length=input_model.limit)

        if not results:
            return QueryMongoRequestsOutput(success=False, message=f"No data found in MongoDB for hashes: {final_hashes}")

        df = pd.DataFrame(results)

        try:
            AnalysisContext.clear_memory()
        except Exception as e:
            return QueryMongoRequestsOutput(success=False, message=f"Error on clear memory: {e}")

        AnalysisContext.set_mongo_data(
            df=df, 
            source=input_model.traffic_source, 
        )

        return QueryMongoRequestsOutput(
            success=True,
            message=(
                f"SUCCESS: Loaded {len(df)} requests into AnalysisContext.\n"
                f"Sources: {input_model.traffic_source} | Hashes: {len(final_hashes)}\n"
                "Action Required: Delegate to 'Metrics Analyst' agent to run ML inference now."
            ),
            num_requests=len(df)
        )

    except Exception as e:
        return QueryMongoRequestsOutput(success=False, message=f"Error loading data from Mongo: {str(e)}")

@tool(args_schema=QuerySqlCampaignsInput)
async def query_sql_campaigns(
    **kwargs
) -> QuerySqlCampaignsOutput:
    """    
    Use this tool when the user asks to analyze traffic but doesn't provide a specific hash.
    It returns a list of available campaign hashes and their metadata.
    
    Returns: A list of campaigns hashes to use query_mongo_requests to search for their data.
    """
    input_model = QuerySqlCampaignsInput(**kwargs)
    try:
        campaigns = await campaign_service.fetch_recent_active_campaigns(
            traffic_source=input_model.traffic_source,
            limit=input_model.limit
        )
        
        if not campaigns:
            return QuerySqlCampaignsOutput(success=False, message=f"No active campaigns found for source: {input_model.traffic_source}.")

        return QuerySqlCampaignsOutput(success=True, campaigns=campaigns)

    except Exception as e:
        return QuerySqlCampaignsOutput(success=False, message=f"Error querying SQL: {str(e)}")
