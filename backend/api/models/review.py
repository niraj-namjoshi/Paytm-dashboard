"""
Pydantic models for review analysis requests and responses
"""

from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class ReviewInput(BaseModel):
    """Single review input model"""
    id: int
    comment: str
    rating: int
    location: str

class ReviewsAnalysisRequest(BaseModel):
    """Request model for reviews analysis"""
    reviews: List[ReviewInput]


