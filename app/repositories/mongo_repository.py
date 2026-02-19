import asyncio
from app.config.mongo import mongo_collection

class MongoRepository:
      def __init__(self, collection):
            self.collection = collection

      async def create_indexes(self):
        print("Criando índices do repositório...")
        await self.collection.create_index([
            ("metadata.site", 1),
            ("decision", 1),
        ])

      
      async def get_recent_requests_by_hashes(self, hashes: list[str], limit: int = 1000) -> list[dict]:

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
      

      async def get_training_sample_by_hashes(self, hashes: list[str], limit_each: int = 10000) -> list[dict]:

            projection = {
                  "headers": True,
                  "request": True,
                  "decision": True,
                  '_id': True,
                  "datetime": True
            } 

            query_bots = {
                  "metadata.site": {"$in": hashes},
                  "decision": {"$in": ["bots"]}
            } 

            
            query_unsafe = {
                  "metadata.site": {"$in": hashes},
                  "decision": {"$in": ["unsafe"]}
            }

            results = await asyncio.gather(
                  self.collection.find(query_bots, projection)
                  .limit(limit_each)
                  .sort("datetime", -1)
                  .to_list(),

                  self.collection.find(query_unsafe, projection)
                  .limit(limit_each)
                  .sort("datetime", -1)
                  .to_list(),
            )

            bots_list = results[0]
            unsafe_list = results[1]

            min_count = min(len(bots_list), len(unsafe_list))
            if min_count == 0:
                  return []
            
            final_bots = bots_list[:min_count]
            final_unsafe = unsafe_list[:min_count]

            return final_bots + final_unsafe



