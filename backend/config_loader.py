"""
Configuration loader for the dashboard backend
"""

import yaml
import os
from typing import Dict, Any, List

class ConfigLoader:
    """Load and manage configuration from YAML file"""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            # Default to config.yaml in the root directory
            self.config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
        else:
            self.config_path = config_path
        
        self._config = None
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                self._config = yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        Example: get('security.jwt.secret_key')
        """
        keys = key_path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    # Convenience methods for commonly used config sections
    
    def get_users(self) -> Dict[str, Dict[str, Any]]:
        """Get all users configuration"""
        users = {}
        
        # FSE Teams
        fse_teams = self.get('users.fse_teams', [])
        for team in fse_teams:
            users[team['username']] = {
                'password': team['password'],
                'type': team['type'],
                'department': team['department']
            }
        
        # Area Managers
        managers = self.get('users.area_managers', [])
        for manager in managers:
            users[manager['username']] = {
                'password': manager['password'],
                'type': manager['type'],
                'location': manager['location']
            }
        
        return users
    
    def get_team_categories(self) -> List[str]:
        """Get team categories for analysis"""
        return self.get('analysis.teams.categories', ['UX', 'Dev', 'Payments', 'Other'])
    
    def get_team_features(self, team_type: str) -> List[str]:
        """Get features for specific team type"""
        if team_type.lower() == 'manager':
            return self.get('team_features.managers', [])
        else:
            team_key = f"team_features.{team_type.lower()}_team"
            return self.get(team_key, [])
    
    def get_jwt_config(self) -> Dict[str, Any]:
        """Get JWT configuration"""
        return {
            'secret_key': self.get('security.jwt.secret_key'),
            'algorithm': self.get('security.jwt.algorithm', 'HS256'),
            'expire_minutes': self.get('security.jwt.access_token_expire_minutes', 30)
        }
    
    def get_ai_config(self) -> Dict[str, Any]:
        """Get AI/ML configuration"""
        return {
            'gemini_model': self.get('ai.google_gemini.model', 'gemini-2.0-flash'),
            'gemini_api_key_env': self.get('ai.google_gemini.api_key_env', 'api_k'),
            'sentiment_model': self.get('ai.sentiment_analysis.model', 'cardiffnlp/twitter-roberta-base-sentiment'),
            'embeddings_model': self.get('ai.embeddings.model', 'all-MiniLM-L6-v2'),
            'max_clusters': self.get('ai.clustering.max_clusters', 20),
            'random_state': self.get('ai.clustering.random_state', 42)
        }
    
    def get_analysis_config(self) -> Dict[str, Any]:
        """Get analysis configuration"""
        return {
            'sentiment_categories': self.get('analysis.sentiment.categories', ['positive', 'neutral', 'negative']),
            'nps_promoters': self.get('analysis.nps.promoters', [4, 5]),
            'nps_neutrals': self.get('analysis.nps.neutrals', [3]),
            'nps_detractors': self.get('analysis.nps.detractors', [1, 2]),
            'team_categories': self.get_team_categories(),
            'similarity_threshold': self.get('analysis.clustering.similarity_threshold', 0.6),
            'severity_thresholds': self.get('analysis.clustering.severity_thresholds', {
                'low': [1, 5],
                'medium': [6, 15],
                'high': [16, 999]
            }),
            'top_representatives': self.get('analysis.clustering.top_representatives', 6)
        }
    
    def get_refresh_config(self) -> Dict[str, Any]:
        """Get refresh configuration"""
        return {
            'incremental_interval': self.get('refresh.incremental_interval', 120),
            'full_refresh_interval': self.get('refresh.full_refresh_interval', 1800),
            'background_enabled': self.get('refresh.background_enabled', True)
        }
    
    def get_prompts(self) -> Dict[str, str]:
        """Get LLM prompts"""
        return {
            'cluster_summary': self.get('prompts.cluster_summary', ''),
            'team_tagging': self.get('prompts.team_tagging', '')
        }
    
    def get_server_config(self) -> Dict[str, Any]:
        """Get server configuration"""
        return {
            'backend_host': self.get('server.backend.host', '0.0.0.0'),
            'backend_port': self.get('server.backend.port', 8000),
            'cors_origins': self.get('security.cors.allow_origins', ['http://localhost:8501'])
        }
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data configuration"""
        return {
            'reviews_file': self.get('data.reviews_file', 'Data/reviews.json'),
            'auto_analyze': self.get('data.auto_analyze_on_startup', True),
            'cache_results': self.get('data.cache_analysis_results', True)
        }

# Global config instance
config = ConfigLoader()
