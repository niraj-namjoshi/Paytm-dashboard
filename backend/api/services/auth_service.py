"""
Authentication service for user management and validation
"""

from config_loader import config

class AuthService:
    """Service class for authentication and user management"""
    
    @classmethod
    def get_users(cls):
        """Get users from configuration"""
        return config.get_users()
    
    @classmethod
    def validate_credentials(cls, username: str, password: str) -> bool:
        """Validate user credentials"""
        users = cls.get_users()
        user = users.get(username)
        if user and user["password"] == password:
            return True
        return False
    
    @classmethod
    def get_user_data(cls, username: str) -> dict:
        """Get user data by username"""
        users = cls.get_users()
        user = users.get(username)
        if user:
            user_data = user.copy()
            user_data.pop("password", None)  # Remove password from response
            return user_data
        return None
    
    @classmethod
    def get_department_features(cls, department: str) -> list:
        """Get features available for specific department"""
        # Map department names to config keys
        dept_mapping = {
            "UX Team": "ux",
            "Payment Team": "payment", 
            "Dev Team": "dev"
        }
        team_key = dept_mapping.get(department)
        if team_key:
            return config.get_team_features(team_key)
        return []
    
    @classmethod
    def get_manager_features(cls) -> list:
        """Get features available for area managers"""
        return config.get_team_features('manager')
