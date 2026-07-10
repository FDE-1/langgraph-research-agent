import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

client_path = Path(os.getenv("client_path") or "workspace/chroma")
collection_name = os.getenv("collection_name") or "agent_memory"
openapi_key = os.getenv("OPENAI_API_KEY")
workspace = Path(os.getenv("workspace") or "workspace").resolve()
