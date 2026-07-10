"""No environment setup needed: importing the package builds no external resources.

Agent's Chroma collection, SQLite checkpointer and OpenAI client are constructed in
`Agent.__init__` and injectable, so tests never need an API key or a writable CWD.
See `test_import_side_effects.py`, which locks that property down.
"""
