import os

# Importing agent.py builds the OpenAI embedding function at module load,
# which requires a non-empty key. Tests don't hit the network, so a dummy
# value is enough to let collection/import succeed without a real API key.
os.environ.setdefault("OPENAI_API_KEY", "sk-test-dummy")
