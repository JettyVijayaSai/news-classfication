"""Launcher script for the Fake News Detection API service."""

import logging
import sys
from pathlib import Path
import uvicorn

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import AppConfig
from src.api.app import app

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    print(f"Starting Fake News Classification API on http://{AppConfig.host}:{AppConfig.port}...")
    uvicorn.run(
        app,
        host=AppConfig.host,
        port=AppConfig.port,
        log_level="info"
    )


if __name__ == "__main__":
    main()
