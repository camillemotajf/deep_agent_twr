class MongoRepository:
      def __init__(self, collection):
            self.collection = collection

      
      async def get_recent_requests_by_hashes(self, hashes: list[str], limit: int = 1000) -> list[dict]:

            await self.collection.create_index("metadata.site")

            projection = {
                  "headers": True,
                  "request": True,
                  "decision": True,
                  '_id': True,
                  "datetime": True
            }      

            cursor = (
                        self.collection.find(
                              {
                                    "metadata.site": {"$in": hashes},
                                    "decision": {"$in": ["bots", "unsafe"]}
                              },
                              projection=projection
                        )
                        .sort("datetime", -1)
                        .limit(limit)
                  )

            return cursor