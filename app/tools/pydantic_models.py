from pydantic import BaseModel, Field
from typing import List, Optional

class QueryMongoRequestsInput(BaseModel):
    traffic_source: str = Field(..., description="The source of the traffic, e.g., 'google', 'facebook'.")
    hash: Optional[str] = Field(None, description="A single campaign hash.")
    hashes: Optional[List[str]] = Field(None, description="A list of campaign hashes.")
    limit: int = Field(1000, description="The maximum number of requests to return.")

class QueryMongoRequestsOutput(BaseModel):
    success: bool
    message: str
    num_requests: int = 0

class QuerySqlCampaignsInput(BaseModel):
    traffic_source: Optional[str] = Field(None, description="The source of the traffic, e.g., 'google', 'facebook'.")
    limit: int = Field(10, description="The maximum number of campaigns to return.")

class QuerySqlCampaignsOutput(BaseModel):
    success: bool
    campaigns: Optional[List[str]] = None
    message: Optional[str] = None

class TrafficSourceByCampaignInput(BaseModel):
    hash: str = Field(..., description="The campaign hash.")

class TrafficSourceByCampaignOutput(BaseModel):
    success: bool
    traffic_source: Optional[str] = None
    message: Optional[str] = None

class QueryRequestsForTrainingInput(BaseModel):
    traffic_source: Optional[str] = Field(None, description="The source of the traffic, e.g., 'google', 'facebook'.")
    hashes: Optional[List[str]] = Field(None, description="A list of campaign hashes.")
    limit: int = Field(10000, description="The maximum number of requests to return.")

class QueryRequestsForTrainingOutput(BaseModel):
    success: bool
    results: Optional[List[dict]] = None
    message: Optional[str] = None
   