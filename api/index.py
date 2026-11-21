"""
api/index.py - Serverless entrypoint for Vercel deployment.
Exposes the FastAPI ASGI application as 'app'.
"""
import os
import sys
import traceback

# Ensure the root project directory is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from web.app import app
except Exception as e:
    tb = traceback.format_exc()
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse

    app = FastAPI(title="The Looker Diagnostic Mode")

    @app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE"])
    async def import_error_fallback(full_path: str = ""):
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>The Looker — Startup Diagnostics</title>
    <style>
        body {{ background: #0f172a; color: #f87171; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; padding: 32px; }}
        .card {{ background: #1e293b; border: 1px solid #dc2626; border-radius: 8px; padding: 24px; max-width: 900px; margin: 0 auto; }}
        h2 {{ color: #ef4444; margin-top: 0; }}
        pre {{ background: #0b0f19; padding: 16px; border-radius: 6px; overflow-x: auto; color: #fca5a5; line-height: 1.5; }}
    </style>
</head>
<body>
    <div class="card">
        <h2>Vercel Cold-Start Import Failure</h2>
        <p>The Python ASGI engine encountered an unhandled exception while loading:</p>
        <pre>{tb}</pre>
    </div>
</body>
</html>"""
        return HTMLResponse(content=html, status_code=500)
