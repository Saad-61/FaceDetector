"""
main.py — Run Face Detector with ngrok tunnel.

Usage:
  PowerShell:
    $env:NGROK_AUTHTOKEN="your_token_here"; python main.py

  Command Prompt (cmd):
    set NGROK_AUTHTOKEN=your_token_here && python main.py

  Bash / Linux / macOS:
    NGROK_AUTHTOKEN=your_token_here python main.py
"""

import os
import sys

# Ensure the project root is on the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import ngrok
from app.server import app

PORT = int(os.environ.get("PORT", 5000))
DEFAULT_DOMAIN = "alongside-vagueness-unfitted.ngrok-free.dev"
DOMAIN = os.environ.get("NGROK_DOMAIN", DEFAULT_DOMAIN)


def connect_ngrok(port: int = PORT, domain: str = DOMAIN):
    """
    Establish ngrok tunnel using authtoken from environment.
    """
    try:
        kwargs = {"authtoken_from_env": True}
        if domain:
            kwargs["domain"] = domain

        forwarder = ngrok.forward(f"localhost:{port}", **kwargs)
        url = forwarder.url()
        print()
        print("=" * 60)
        print("  ✓ ngrok tunnel established successfully!")
        print(f"  Public URL : {url}")
        print(f"  Local URL  : http://localhost:{port}")
        print("  Frontend and /detect endpoint are ready for testing.")
        print("=" * 60)
        print()
        return forwarder
    except Exception as e:
        print()
        print("=" * 60)
        print("  [ngrok] Could not establish tunnel:")
        print(f"  {e}")
        print()
        print("  To run with ngrok, set your auth token first:")
        print("    PowerShell: $env:NGROK_AUTHTOKEN = 'your_token'")
        print("    CMD:        set NGROK_AUTHTOKEN=your_token")
        print("    Bash:       export NGROK_AUTHTOKEN=your_token")
        print("=" * 60)
        print()
        return None


if __name__ == "__main__":
    # Establish ngrok tunnel
    connect_ngrok(port=PORT, domain=DOMAIN)

    # Start Flask web server
    app.run(
        debug=False,
        host="0.0.0.0",
        port=PORT,
    )
