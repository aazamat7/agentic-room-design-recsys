from typing import List, Optional
from pydantic import BaseModel, Field


class ProductCard(BaseModel):
    title: str = ""
    description: str = ""
    price_text: Optional[str] = None
    product_url: str = ""
    image_url: Optional[str] = None
    source_site: str = ""
    category: Optional[str] = None
    relevance_score: int = Field(default=0, ge=0, le=100)
    why_it_matches: str = ""


class ProductBrowseResult(BaseModel):
    products: List[ProductCard] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class RecommendationOutput(BaseModel):
    interpreted_need: str
    recommended_products: List[ProductCard]
    missing_information: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)