"""
run.py — Entry point for the Face Detector web app.

Usage:
    python run.py

Then open: http://localhost:5000
"""

import sys
import os

# Ensure the project root is on the Python path so imports work
# regardless of which directory you run from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.server import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    # Optional ngrok connection if authtoken is present or --ngrok flag passed
    if os.environ.get("NGROK_AUTHTOKEN") or "--ngrok" in sys.argv:
        try:
            from main import connect_ngrok
            connect_ngrok(port=port)
        except Exception as err:
            print(f"[ngrok] Tunnel failed to start: {err}")

    print()
    print("=" * 50)
    print("  Face Detector")
    print(f"  http://localhost:{port}")
    print("  Press Ctrl+C to stop")
    print("=" * 50)
    print()
    app.run(
        debug=False,      # set True during development for auto-reload
        host="0.0.0.0",   # accessible from local network too
        port=port,
    )

