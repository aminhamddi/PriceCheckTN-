#!/usr/bin/env python3
"""
Script to start Prefect dashboard with correct local configuration
"""

import os
from dotenv import load_dotenv

def main():
    # Load environment variables from .env file
    load_dotenv()

    # Verify the API URL is set correctly
    api_url = os.getenv('PREFECT_API_URL')
    print(f"🔧 PREFECT_API_URL: {api_url}")

    if api_url != 'http://127.0.0.1:4200/api':
        print("❌ PREFECT_API_URL is not set correctly!")
        return 1

    print("✅ Environment configured correctly!")
    print("🌐 Opening Prefect dashboard at: http://127.0.0.1:4200")
    print("📊 You can now access the dashboard in your browser")

    return 0

if __name__ == "__main__":
    exit(main())
