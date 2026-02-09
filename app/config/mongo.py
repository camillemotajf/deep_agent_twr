import os
from motor.motor_asyncio import AsyncIOMotorClient
from app.config.settings import settings

CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_FILE_DIR))
CERT_FILE_PATH = os.path.join(PROJECT_ROOT, "keys", "global-bundle.pem")



mongo_client = AsyncIOMotorClient(
      settings.MONGO_URI,
      tls=True,
      tlsCAFile=CERT_FILE_PATH,
      directConnection=True,
      retryWrites=False,
      authMechanism="SCRAM-SHA-1"
)

mongo_db = mongo_client[settings.DB_NAME]
mongo_collection = mongo_db[settings.COL_REQUEST]
