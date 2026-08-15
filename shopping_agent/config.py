import os
from dotenv import load_dotenv

load_dotenv(override=True)


class Settings:
    GOOGLE_CLOUD_PROJECT: str = os.environ["GOOGLE_CLOUD_PROJECT"]
    GOOGLE_CLOUD_LOCATION: str = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    GCS_STAGING_BUCKET: str = os.environ["GCS_STAGING_BUCKET"]
    
    BROWSERBASE_API_KEY: str = os.environ["BROWSERBASE_API_KEY"]
    GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    DEFAULT_COUNTRY: str = os.environ.get("DEFAULT_COUNTRY", "US")


settings = Settings()