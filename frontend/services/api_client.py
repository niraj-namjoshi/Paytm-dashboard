"""
API Client for communicating with the backend
"""

import requests
import streamlit as st
from typing import Optional, Dict, Any

class APIClient:
    """Client for backend API communication"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> bool:
        """Check if backend API is accessible"""
        try:
            response = self.session.get(f"{self.base_url}/", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def login(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Login user and get JWT token"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/auth/login",
                json={"username": username, "password": password},
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return None
                
        except requests.exceptions.RequestException as e:
            st.error(f"Connection error: {str(e)}")
            return None
    
    def get_user_info(self, token: str) -> Optional[Dict[str, Any]]:
        """Get current user information"""
        try:
            headers = {"Authorization": f"Bearer {token}"}
            response = self.session.get(
                f"{self.base_url}/api/auth/me",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return None
                
        except requests.exceptions.RequestException:
            return None
    
    def get_dashboard_data(self, token: str) -> Optional[Dict[str, Any]]:
        """Get dashboard data for authenticated user"""
        try:
            headers = {"Authorization": f"Bearer {token}"}
            response = self.session.get(
                f"{self.base_url}/api/dashboard/data",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return None
                
        except requests.exceptions.RequestException:
            return None

    def get_team_data(self, team: str, token: str) -> Optional[Dict[str, Any]]:
        """Get team-specific analysis data"""
        try:
            headers = {"Authorization": f"Bearer {token}"}
            response = self.session.get(
                f"{self.base_url}/api/teams/{team.lower()}",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return None
                
        except requests.exceptions.RequestException:
            return None

    def get_nps_scores(self, token: str) -> Optional[Dict[str, Any]]:
        """Get NPS scores for all teams"""
        try:
            headers = {"Authorization": f"Bearer {token}"}
            response = self.session.get(
                f"{self.base_url}/api/nps/scores",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return None
                
        except requests.exceptions.RequestException:
            return None

    def get_location_data(self, location: str, token: str) -> Optional[Dict[str, Any]]:
        """Get location-specific analysis data"""
        try:
            headers = {"Authorization": f"Bearer {token}"}
            response = self.session.get(
                f"{self.base_url}/api/locations/{location.lower()}",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return None
                
        except requests.exceptions.RequestException:
            return None

    def get_location_nps(self, token: str) -> Optional[Dict[str, Any]]:
        """Get location-wise NPS breakdown"""
        try:
            headers = {"Authorization": f"Bearer {token}"}
            response = self.session.get(
                f"{self.base_url}/api/nps/location",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return None
                
        except requests.exceptions.RequestException:
            return None

    def get_location_sentiment(self, token: str) -> Optional[Dict[str, Any]]:
        """Get location-based sentiment distribution"""
        try:
            headers = {"Authorization": f"Bearer {token}"}
            response = self.session.get(
                f"{self.base_url}/api/sentiment/location",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return None
                
        except requests.exceptions.RequestException:
            return None

# Global API client instance
api_client = APIClient()
