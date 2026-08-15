from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


TaskType = Literal[
    "single_product_search",
    "bundle_recommendation",
    "similar_product_search",
    "gift_recommendation",
    "style_based_recommendation",
]


class UserProfile(BaseModel):
    styles: List[str] = Field(default_factory=list)
    colors: List[str] = Field(default_factory=list)
    materials: List[str] = Field(default_factory=list)
    brands: List[str] = Field(default_factory=list)
    avoid: List[str] = Field(default_factory=list)
    room_or_use_case: Optional[str] = None


class UserConstraints(BaseModel):
    country: str = "US"
    currency: str = "USD"
    total_budget: Optional[float] = None
    required_categories: List[str] = Field(default_factory=list)
    must_have: List[str] = Field(default_factory=list)
    nice_to_have: List[str] = Field(default_factory=list)


class PlannerOutput(BaseModel):
    interpreted_need: str = ""
    task_type: TaskType = "single_product_search"
    user_profile: UserProfile = Field(default_factory=UserProfile)
    constraints: UserConstraints = Field(default_factory=UserConstraints)
    browser_query: str = ""
    search_strategy: str = "web_only"


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


class VerifiedProduct(ProductCard):
    is_valid: bool = False
    verification_issues: List[str] = Field(default_factory=list)


class RankedProduct(VerifiedProduct):
    final_score: float = 0.0
    score_reasons: List[str] = Field(default_factory=list)


class BundleResult(BaseModel):
    products: List[RankedProduct] = Field(default_factory=list)
    missing_categories: List[str] = Field(default_factory=list)
    total_price_text: Optional[str] = None
    notes: List[str] = Field(default_factory=list)


class CriticResult(BaseModel):
    is_valid: bool
    issues: List[str] = Field(default_factory=list)
    final_products: List[RankedProduct] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class RecommendationOutput(BaseModel):
    interpreted_need: str
    recommended_products: List[ProductCard]
    missing_information: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)