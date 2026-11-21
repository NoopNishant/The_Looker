"""
api/index.py - Serverless entrypoint for Vercel deployment.
Exposes the FastAPI ASGI application as 'app'.
"""
import os
import sys

# Ensure the root project directory is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import FastAPI app instance from web.app
from web.app import app
