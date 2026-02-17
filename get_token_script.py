
import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.getcwd())

from app.core.auth_client import ensure_access_token

async def main():
    try:
        token = await ensure_access_token(interactive=False)
        print(token)
    except Exception as e:
        print(f"Error fetching token: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
