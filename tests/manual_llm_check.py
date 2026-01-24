import asyncio

from langchain_core.messages import HumanMessage

from config import get_settings
from pathlib import Path
from utils.llm import get_llm


async def main():

    # Debug: Print loaded settings
    settings = get_settings()
    print("=" * 50)
    print("DEBUG: Checking loaded settings")
    print("=" * 50)
    print(f"DIAL_ENDPOINT: {settings.dial_endpoint}")
    print(f"DIAL_DEPLOYMENT: {settings.dial_deployment}")
    print(
        f"DIAL_API_KEY: {'*' * 10 + settings.dial_api_key[-4:] if settings.dial_api_key else 'NOT SET'}"
    )
    print(f"DIAL_API_VERSION: {settings.dial_api_version}")
    print("=" * 50)

    if not settings.dial_api_key:
        print("ERROR: DIAL_API_KEY is not set!")
        print(f"Looking for .env file in: {Path.cwd()}")
        print(f".env exists: {(Path.cwd() / '.env').exists()}")
        return

    llm = get_llm()
    messages = [[HumanMessage(content="Hello from the AI E-commerce Brain!")]]

    resp = await llm.agenerate(messages)
    print(resp.generations[0][0].text)


if __name__ == "__main__":
    asyncio.run(main())
