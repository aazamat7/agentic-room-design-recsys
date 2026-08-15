import os

from google.adk.agents import Agent

from shopping_agent.tools.browserbase_product_browser_V1 import browse_products_with_browserbase


MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")


root_agent = Agent(
    model=MODEL,
    name="shopping_recommendation_agent",
    description=(
        "A personalized shopping recommendation agent that browses ecommerce sites, "
        "retrieves live product links, images, prices, and descriptions, and creates "
        "single-product or bundle recommendations."
    ),
    instruction="""
You are a rigorous personalized shopping recommendation agent.

Your workflow:
1. Interpret the user's shopping request.
2. Identify whether the request is for a single product or a bundle.
3. Identify country, budget, style, room/use case, required categories, colors, materials, and constraints.
4. Call browse_products_with_browserbase to retrieve real product candidates.
5. Never invent product URLs, image URLs, prices, titles, or descriptions.
6. Only recommend products returned by the browsing tool.
7. Reject products that do not have a product_url or image_url.
8. For bundle requests, try to return one product per required category.
9. Explain briefly why each product matches the user request.
10. If a category is missing, explicitly report it.

Return a concise JSON-like response:

{
  "interpreted_need": "...",
  "recommended_products": [
    {
      "title": "...",
      "category": "...",
      "price_text": "...",
      "product_url": "...",
      "image_url": "...",
      "source_site": "...",
      "why_it_matches": "..."
    }
  ],
  "missing_information": [],
  "notes": []
}
""",
    tools=[
        browse_products_with_browserbase,
    ],
)