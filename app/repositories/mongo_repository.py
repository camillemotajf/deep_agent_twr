import asyncio
from typing import Dict, List

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
                  'ip_api_isp': True,
                  "datetime": True,
                  "ip": True
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
      
      def _make_query(decision_type: List, hashes, only_rule_id):
            query = {
                  "metadata.site": {"$in": hashes},
                  "decision": {"$in": decision_type}
            }

            if only_rule_id:
                  query["rule_id_list"] = {"$exists": True}

            return query
      

      async def get_training_sample_by_hashes(self, hashes: list[str], limit_each: int = 10000, only_rule_id: bool = False) -> List[Dict]:

            projection = {
                 "_id": False,
                  "headers": True,
                  "request": True,
                  "decision": True,
                  "ip_api_isp": True,
                  "datetime": True,
                  "ip": True
            } 

            if only_rule_id:
                 projection["rule_id_list"] = True


            results = await asyncio.gather(
                  self.collection.find(
                       self._make_query(decision_type=["bots", "bot"],
                                         hashes=hashes, 
                                         only_rule_id=only_rule_id
                                    ), 
                        projection=projection
                  )
                  .limit(limit_each)
                  .sort("datetime", -1)
                  .to_list(),

                  self.collection.find(
                       self._make_query(decision_type=["unsafe"],
                                         hashes=hashes, 
                                         only_rule_id=only_rule_id
                                    ), 
                        projection=projection
                  )
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



