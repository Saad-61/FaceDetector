"""
run.py — CLI entry point to launch the Face Detector web application.

Usage:
    python run.py
    python run.py --port 5001 --device cuda:0
"""

import argparse
from app.server import app

def main():
    parser = argparse.ArgumentParser(description="Face Detector Web Application")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind to (default: 5000)")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"  Face Detector running at: http://{args.host}:{args.port}")
    print(f"{'='*55}\n")

    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
