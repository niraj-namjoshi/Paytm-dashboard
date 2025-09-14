"""
Authentication service for user management and validation
"""

class AuthService:
    """Service class for authentication and user management"""
    
    # User credentials database
    USERS = {
        # FSE Teams
        "ux_team": {
            "password": "password123",
            "type": "FSE Team",
            "department": "UX Team"
        },
        "payment_team": {
            "password": "password123",
            "type": "FSE Team",
            "department": "Payment Team"
        },
        "dev_team": {
            "password": "password123",
            "type": "FSE Team",
            "department": "Dev Team"
        },
        
        # Area Managers
        "mumbai_manager": {
            "password": "password123",
            "type": "Area Manager",
            "location": "Mumbai"
        },
        "bangalore_manager": {
            "password": "password123",
            "type": "Area Manager",
            "location": "Bangalore"
        }
    }
    
    @classmethod
    def validate_credentials(cls, username: str, password: str) -> bool:
        """Validate user credentials"""
        user = cls.USERS.get(username)
        if user and user["password"] == password:
            return True
        return False
    
    @classmethod
    def get_user_data(cls, username: str) -> dict:
        """Get user data by username"""
        user = cls.USERS.get(username)
        if user:
            user_data = user.copy()
            user_data.pop("password", None)  # Remove password from response
            return user_data
        return None
    
    @classmethod
    def get_department_features(cls, department: str) -> list:
        """Get features available for specific department"""
        features_map = {
            "UX Team": [
                "User Research Dashboard",
                "Design System Management",
                "Prototype Testing Tools",
                "User Feedback Analytics",
                "A/B Testing Results"
            ],
            "Payment Team": [
                "Transaction Monitoring",
                "Payment Gateway Analytics",
                "Fraud Detection Dashboard",
                "Revenue Tracking",
                "Refund Management"
            ],
            "Dev Team": [
                "Code Repository Access",
                "Build Pipeline Status",
                "Bug Tracking System",
                "Performance Monitoring",
                "Deployment Dashboard"
            ]
        }
        return features_map.get(department, [])
    
    @classmethod
    def get_manager_features(cls) -> list:
        """Get features available for area managers"""
        return [
            "Team Performance Overview",
            "Resource Allocation Dashboard",
            "Project Timeline Management",
            "Budget Tracking",
            "Employee Management",
            "Regional Analytics",
            "Cross-team Collaboration Tools",
            "Strategic Planning Dashboard"
        ]
