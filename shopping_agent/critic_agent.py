import os

from google.adk.agents.llm_agent import LlmAgent

from shopping_agent.tools.recommendation_tools import critique_recommendation


MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


critic_agent = LlmAgent(
    name="CriticAgent",
    model=MODEL,
    description="Final guardrail agent that checks product URLs, images, duplicates, and category coverage.",
    instruction="""
You are the final Critic Agent.

Planner JSON:
{planner_output}

Bundle JSON:
{bundle_output}

Call critique_recommendation with:
- planner_json = planner JSON
- bundle_json = bundle JSON

Then return a final concise JSON object:

{
  "is_valid": true,
  "issues": [],
  "recommended_products": [
    {
      "title": "...",
      "category": "...",
      "price_text": "...",
      "product_url": "...",
      "image_url": "...",
      "source_site": "...",
      "why_it_matches": "...",
      "final_score": 0
    }
  ],
  "missing_categories": [],
  "notes": []
}

Rules:
- Only include products from critique_recommendation.final_products.
- Do not invent product URLs, image URLs, prices, or titles.
- If issues exist, include them clearly.
- If categories are missing, report them.
""",
    tools=[critique_recommendation],
    output_key="final_recommendation",
)