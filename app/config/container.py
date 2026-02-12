from app.repositories.sql_repository import CampaignRepository
from app.services.campaign_service import CampaignService
from app.config.database import AsyncSessionLocal

from app.config.mongo import mongo_collection
from app.repositories.mongo_repository import MongoRepository
from app.services.request_service import RequestService


campaign_repository = CampaignRepository(
    session_factory=AsyncSessionLocal
)

campaign_service = CampaignService(
    repository=campaign_repository
)

mongo_request_repository = MongoRepository(
    collection=mongo_collection
)

request_service = RequestService(
    repository=mongo_request_repository
)


