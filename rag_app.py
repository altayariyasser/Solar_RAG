"""Compatibility entrypoint.

New deployments should use streamlit_app.py. Existing deployments configured
for rag_app.py continue to work through this import.
"""

from streamlit_app import *  # noqa: F401,F403
