import os
import json
import yaml
from typing import Dict, Any, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_CONFIG = {
    'app': {
        'title': 'Data Analysis Dashboard',
        'version': '1.0.0',
        'debug': False,
        'max_file_size_mb': 100,
        'supported_formats': ['csv', 'xlsx', 'json'],
        'default_chart_height': 500,
        'max_rows_display': 1000
    },
    'data_processing': {
        'default_missing_strategy': 'mean',
        'outlier_detection_method': 'iqr',
        'outlier_threshold': 1.5,
        'correlation_threshold': 0.7,
        'max_categories_pie_chart': 10,
        'normalization_method': 'standard',
        'sample_size_for_large_datasets': 5000
    },
    'visualization': {
        'default_theme': 'plotly_white',
        'color_palette': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
        'chart_width': 800,
        'chart_height': 500,
        'dpi': 100,
        'font_size': 12,
        'show_grid': True,
        'opacity': 0.7
    },
    'export': {
        'default_format': 'csv',
        'backup_enabled': True,
        'backup_directory': 'backups',
        'report_template': 'default',
        'include_statistics': True,
        'compress_output': False
    },
    'performance': {
        'chunk_size': 10000,
        'memory_limit_mb': 512,
        'cache_enabled': True,
        'parallel_processing': False,
        'max_workers': 4
    },
    'security': {
        'allowed_file_types': ['.csv', '.xlsx', '.json', '.txt'],
        'max_upload_size': 50,  # MB
        'sanitize_input': True,
        'log_user_actions': True
    }
}

class ConfigManager:
    def __init__(self, config_file=None):
        self.config_file = config_file or 'config.json'
        self.config = DEFAULT_CONFIG.copy()
        self.load_config()
    
    def load_config(self):
        """Load configuration from file"""
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    if self.config_file.endswith('.yaml') or self.config_file.endswith('.yml'):
                        file_config = yaml.safe_load(f)
                    else:
                        file_config = json.load(f)
                
                # Merge with default config
                self.config = self._merge_configs(self.config, file_config)
                logger.info(f"Configuration loaded from {self.config_file}")
                
            except Exception as e:
                logger.error(f"Error loading config file {self.config_file}: {str(e)}")
                logger.info("Using default configuration")
        else:
            logger.info(f"Config file {self.config_file} not found. Using default configuration.")
            self.save_config()  # Create default config file
    
    def save_config(self):
        """Save current configuration to file"""
        
        try:
            with open(self.config_file, 'w') as f:
                if self.config_file.endswith('.yaml') or self.config_file.endswith('.yml'):
                    yaml.dump(self.config, f, default_flow_style=False, indent=2)
                else:
                    json.dump(self.config, f, indent=2)
            
            logger.info(f"Configuration saved to {self.config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving config file: {str(e)}")
            return False
    
    def get(self, key_path, default=None):
        """Get configuration value using dot notation"""
        
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path, value):
        """Set configuration value using dot notation"""
        
        keys = key_path.split('.')
        config_ref = self.config
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in config_ref:
                config_ref[key] = {}
            config_ref = config_ref[key]
        
        # Set the value
        config_ref[keys[-1]] = value
        logger.info(f"Configuration updated: {key_path} = {value}")
    
    def _merge_configs(self, default_config, file_config):
        """Recursively merge configurations"""
        
        merged = default_config.copy()
        
        for key, value in file_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def validate_config(self):
        """Validate configuration values"""
        
        issues = []
        
        # Check required sections
        required_sections = ['app', 'data_processing', 'visualization', 'export']
        for section in required_sections:
            if section not in self.config:
                issues.append(f"Missing required configuration section: {section}")
        
        # Validate specific values
        max_file_size = self.get('app.max_file_size_mb', 0)
        if max_file_size <= 0 or max_file_size > 1000:
            issues.append("app.max_file_size_mb must be between 1 and 1000")
        
        correlation_threshold = self.get('data_processing.correlation_threshold', 0)
        if correlation_threshold < 0 or correlation_threshold > 1:
            issues.append("data_processing.correlation_threshold must be between 0 and 1")
        
        chunk_size = self.get('performance.chunk_size', 0)
        if chunk_size <= 0:
            issues.append("performance.chunk_size must be greater than 0")
        
        # Validate color palette
        colors = self.get('visualization.color_palette', [])
        if not colors or len(colors) < 3:
            issues.append("visualization.color_palette must contain at least 3 colors")
        
        return issues
    
    def reset_to_defaults(self):
        """Reset configuration to default values"""
        
        self.config = DEFAULT_CONFIG.copy()
        logger.info("Configuration reset to defaults")

# Global configuration instance
config_manager = ConfigManager()

# Configuration constants for easy access
APP_TITLE = config_manager.get('app.title', 'Data Analysis Dashboard')
MAX_FILE_SIZE_MB = config_manager.get('app.max_file_size_mb', 100)
DEFAULT_CHART_HEIGHT = config_manager.get('app.default_chart_height', 500)
SUPPORTED_FORMATS = config_manager.get('app.supported_formats', ['csv', 'xlsx', 'json'])

MISSING_STRATEGY = config_manager.get('data_processing.default_missing_strategy', 'mean')
OUTLIER_THRESHOLD = config_manager.get('data_processing.outlier_threshold', 1.5)
CORRELATION_THRESHOLD = config_manager.get('data_processing.correlation_threshold', 0.7)
SAMPLE_SIZE_LARGE_DATASETS = config_manager.get('data_processing.sample_size_for_large_datasets', 5000)

CHART_THEME = config_manager.get('visualization.default_theme', 'plotly_white')
COLOR_PALETTE = config_manager.get('visualization.color_palette', ['#1f77b4', '#ff7f0e', '#2ca02c'])
CHART_WIDTH = config_manager.get('visualization.chart_width', 800)
CHART_HEIGHT = config_manager.get('visualization.chart_height', 500)

BACKUP_ENABLED = config_manager.get('export.backup_enabled', True)
BACKUP_DIRECTORY = config_manager.get('export.backup_directory', 'backups')
DEFAULT_EXPORT_FORMAT = config_manager.get('export.default_format', 'csv')

CHUNK_SIZE = config_manager.get('performance.chunk_size', 10000)
MEMORY_LIMIT_MB = config_manager.get('performance.memory_limit_mb', 512)
CACHE_ENABLED = config_manager.get('performance.cache_enabled', True)

ALLOWED_FILE_TYPES = config_manager.get('security.allowed_file_types', ['.csv', '.xlsx', '.json'])
MAX_UPLOAD_SIZE_MB = config_manager.get('security.max_upload_size', 50)

# Environment-specific overrides
if os.getenv('DEBUG') == 'true':
    config_manager.set('app.debug', True)

if os.getenv('MAX_FILE_SIZE'):
    try:
        max_size = int(os.getenv('MAX_FILE_SIZE'))
        config_manager.set('app.max_file_size_mb', max_size)
    except ValueError:
        logger.warning("Invalid MAX_FILE_SIZE environment variable")

# Development vs Production settings
if os.getenv('ENVIRONMENT') == 'production':
    config_manager.set('app.debug', False)
    config_manager.set('security.log_user_actions', True)
    config_manager.set('performance.cache_enabled', True)
elif os.getenv('ENVIRONMENT') == 'development':
    config_manager.set('app.debug', True)
    config_manager.set('security.log_user_actions', False)

def initialize_directories():
    """Initialize required directories based on configuration"""
    
    directories = [
        config_manager.get('export.backup_directory', 'backups'),
        'logs',
        'temp',
        'exports'
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
                logger.info(f"Created directory: {directory}")
            except Exception as e:
                logger.error(f"Failed to create directory {directory}: {str(e)}")

# Initialize directories on import
initialize_directories()

# Validate configuration on startup
validation_issues = config_manager.validate_config()
if validation_issues:
    logger.warning("Configuration validation issues found:")
    for issue in validation_issues:
        logger.warning(f"  - {issue}")
else:
    logger.info("Configuration validation passed")