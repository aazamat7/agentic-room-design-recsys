import os

from dotenv import load_dotenv

# Load .env before ADK/model initialization.
load_dotenv()

# Force Vertex AI / ADC mode.
# This prevents ADK from trying to use the Gemini Developer API key path.
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")

PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

if not PROJECT_ID:
    raise RuntimeError(
        "Missing GOOGLE_CLOUD_PROJECT. "
        "Set it in .env or export it before running ADK."
    )

os.environ.setdefault("GOOGLE_CLOUD_LOCATION", LOCATION)

from google.adk.agents.llm_agent import LlmAgent

from shopping_agent.tools.recommendation_tools import verify_product_cards


MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


verification_agent = LlmAgent(
    name="ProductVerificationAgent",
    model=MODEL,
    description="Verifies product cards and removes invalid product pages.",
    instruction="""
You are the Product Verification Agent.

Input product candidates:
{web_discovery_output}

Call verify_product_cards with product_candidates_json equal to the input product candidates.

Return only the tool result JSON.

Do not invent, repair, or add products. Verification must be conservative.
""",
    tools=[verify_product_cards],
    output_key="verification_output",
)