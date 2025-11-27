import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
LLM_API = os.getenv("LLM_API")
EMBED_MODEL = os.getenv("EMBED_MODEL")
LM_STUDIO_URL = os.getenv("LM_STUDIO_URL")