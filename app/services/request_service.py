from app.repositories.mongo_repository import MongoRepository

class RequestService:
    def __init__(self, repository: MongoRepository):
        self.repository = repository

    def fetch_recent_flagged_requests(
        self,
        hashes: list[str],
        limit: int = 1000
    ) -> list[dict]:
        """
        Business rules:
        - Requires at least one hash
        - Returns only bot or unsafe decisions
        """
        if not hashes:
            raise ValueError("At least one campaign hash is required")
        
        results =  self.repository.get_recent_requests_by_hashes(
            hashes=hashes,
            limit=limit
        )
        return results