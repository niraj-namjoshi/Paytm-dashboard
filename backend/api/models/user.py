"""
Pydantic models for user data and API requests/responses
"""

from pydantic import BaseModel
from typing import Optional, Dict, List, Any

class LoginRequest(BaseModel):
    """Login request model"""
    username: str
    password: str

class User(BaseModel):
    """User model"""
    username: str
    user_type: str
    department: Optional[str] = None
    location: Optional[str] = None

class LoginResponse(BaseModel):
    """Login response model"""
    access_token: str
    token_type: str
    user_data: Dict[str, Any]

class DashboardData(BaseModel):
    """Dashboard data model"""
    metrics: Dict[str, Any]
    features: List[str]
    user_info: Dict[str, Any]
