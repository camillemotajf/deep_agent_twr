import asyncio
from typing import Dict, List
from datetime import datetime

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
      
      def _make_query(self, decision_type: List, hashes, only_rule_id):
            query = {
                  "metadata.site": {"$in": hashes},
                  "decision": {"$in": decision_type}
            }

            if only_rule_id:
                  query["rule_id_list"] = {"$exists": True}

            return query
      

      async def get_training_sample_by_hashes(
            self,
            hashes: list[str],
            start: datetime | None = None,
            end: datetime | None = None,
            limit_each: int | None = None,
            only_rule_id: bool = False
      ) -> List[Dict]:

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

            def build_query(types):
                  query = self._make_query(types, hashes, only_rule_id)

                  if start or end:
                        query["datetime"] = {}
                        if start:
                              query["datetime"]["$gte"] = start
                        if end:
                              query["datetime"]["$lte"] = end

                  return query

            async def fetch(types):
                  cursor = self.collection.find(
                        build_query(types),
                        projection=projection
                  ).sort("datetime", -1)

                  if limit_each and not (start or end):
                        cursor = cursor.limit(limit_each)


                  return await cursor.to_list(length=None)

            bots_list, unsafe_list = await asyncio.gather(
                  fetch(["bots", "bot"]),
                  fetch(["unsafe"])
            )


            if limit_each and not (start or end):
                  min_count = min(len(bots_list), len(unsafe_list))
                  if min_count == 0:
                        return []

                  bots_list = bots_list[:min_count]
                  unsafe_list = unsafe_list[:min_count]

            return bots_list + unsafe_list



