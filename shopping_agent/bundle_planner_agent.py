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

from shopping_agent.tools.recommendation_tools import build_recommendation_bundle


MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


bundle_planner_agent = LlmAgent(
    name="BundlePlannerAgent",
    model=MODEL,
    description="Builds a bundle from ranked products, one product per required category where possible.",
    instruction="""
You are the Bundle Planner Agent.

Planner JSON:
{planner_output}

Ranked products JSON:
{reranking_output}

Call build_recommendation_bundle with:
- planner_json = planner JSON
- ranked_products_json = ranked products JSON
- max_products = 8

Return only the tool result JSON.

Rules:
- For bundle requests, prefer one product per required category.
- If a category cannot be filled, preserve missing_categories.
- Do not invent replacement products.
""",
    tools=[build_recommendation_bundle],
    output_key="bundle_output",
)