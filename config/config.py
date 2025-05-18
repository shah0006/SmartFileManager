"""
Configuration Manager Module.

This module handles loading, saving, and providing access to
configuration settings for SmartFileManager.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

# Setup logging
logger = logging.getLogger(__name__)

class ConfigManager:
    """Configuration manager for SmartFileManager."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_file: Path to the configuration file (optional)
        """
        # Set default config file location if not provided
        if config_file is None:
            # Use the config directory in the project
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.config_file = os.path.join(base_dir, "config", "settings.json")
        else:
            self.config_file = config_file
        
        # Load default configuration
        self.config = self._get_default_config()
        
        # Load user configuration if it exists
        if os.path.exists(self.config_file):
            self._load_config()
        else:
            # Save default configuration
            self._save_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get the default configuration settings.
        
        Returns:
            Default configuration dictionary
        """
        return {
            # Core settings
            "core": {
                "to_delete_base_dir": "{project_dir}/to_delete",
                "log_level": "INFO",
                "backup_enabled": True,
                "progress_bars": True,
                "async_processing": True
            },
            
            # LLM settings
            "llm": {
                "provider": "auto",  # Options: "auto", "transformers", "ollama", "none"
                "model": "auto",
                "max_tokens": 500,
                "temperature": 0.2,
                "use_cache": True
            },
            
            # Book module settings
            "books": {
                "organize_by_author": True,
                "organize_by_year": False,
                "organize_by_publisher": False,
                "organize_by_language": False,
                "create_alphabetical_dirs": True,
                "naming_template": "{author}_{year}_{title}",
                "extensions": [".epub", ".mobi", ".azw", ".azw3", ".pdf", ".txt"]
            },
            
            # Academic module settings
            "academic": {
                "naming_template": "{authors}_{year}_{title}",
                "use_llm_extraction": True,
                "include_abstract": False,
                "include_doi": True,
                "zotero_comparison_enabled": True,
                "pdf_metadata": {
                    "ai_override_single_word_titles": True,
                    "use_organization_as_author": True,
                    "max_authors_in_filename": 2,
                    "hyphenate_isbn": True,
                    "detailed_logging": True
                }
            },
            
            # Medical module settings
            "medical": {
                "privacy_level": "high",  # Options: "low", "medium", "high"
                "anonymize_filenames": True,
                "organize_by_doc_type": True,
                "organize_by_date": True,
                "organize_by_facility": False,
                "track_operations": True
            },
            
            # General file settings
            "general": {
                "organize_by_type": True,
                "organize_by_date": False,
                "organize_by_content": True,
                "handle_downloads": True,
                "clean_empty_dirs": True,
                "group_related_files": True,
                "exclude_extensions": [".ini", ".lnk", ".url", ".tmp", ".temp", ".log", ".sys"],
                "prevent_circular_moves": True
            },
            
            # User interface settings
            "ui": {
                "confirm_operations": True,
                "detailed_logging": True,
                "show_preview": True
            }
        }
    
    def _load_config(self):
        """Load configuration from file."""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
            
            # Update our configuration with user settings
            self._update_nested_dict(self.config, user_config)
            
            logger.info(f"Loaded configuration from {self.config_file}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
    
    def _save_config(self):
        """Save configuration to file."""
        try:
            # Create parent directory if it doesn't exist
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4)
            
            logger.info(f"Saved configuration to {self.config_file}")
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def _update_nested_dict(self, base_dict: Dict, update_dict: Dict):
        """
        Update a nested dictionary without completely overwriting sections.
        Only updates keys that exist in update_dict.
        
        Args:
            base_dict: The base dictionary to update
            update_dict: The dictionary with updated values
        """
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                # Recursively update nested dictionaries
                self._update_nested_dict(base_dict[key], value)
            else:
                # Update or add the key-value pair
                base_dict[key] = value
    
    def get_config(self, section: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the configuration or a specific section.
        
        Args:
            section: Section name to retrieve (optional)
            
        Returns:
            Configuration dictionary
        """
        if section is not None:
            return self.config.get(section, {})
        return self.config
    
    def update_config(self, section: str, settings: Dict[str, Any], save: bool = True):
        """
        Update a section of the configuration.
        
        Args:
            section: Section name to update
            settings: New settings dictionary
            save: Whether to save to file after updating
        """
        # Create section if it doesn't exist
        if section not in self.config:
            self.config[section] = {}
        
        # Update the section
        self._update_nested_dict(self.config[section], settings)
        
        # Save configuration if requested
        if save:
            self._save_config()
    
    def get_setting(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get a specific setting value.
        
        Args:
            section: Section name
            key: Setting key
            default: Default value if not found
            
        Returns:
            Setting value
        """
        if section in self.config and key in self.config[section]:
            return self.config[section][key]
        return default
    
    def set_setting(self, section: str, key: str, value: Any, save: bool = True):
        """
        Set a specific setting value.
        
        Args:
            section: Section name
            key: Setting key
            value: New setting value
            save: Whether to save to file after updating
        """
        # Create section if it doesn't exist
        if section not in self.config:
            self.config[section] = {}
        
        # Update the setting
        self.config[section][key] = value
        
        # Save configuration if requested
        if save:
            self._save_config()
    
    def reset_to_defaults(self, save: bool = True):
        """
        Reset configuration to default values.
        
        Args:
            save: Whether to save to file after resetting
        """
        self.config = self._get_default_config()
        
        # Save configuration if requested
        if save:
            self._save_config()
    
    def get_to_delete_dir(self, category: str) -> str:
        """
        Get the appropriate 'to_delete' directory for a category.
        
        Args:
            category: Category name (e.g., 'books', 'academic', 'medical')
            
        Returns:
            Path to the to_delete directory
        """
        # Get base to_delete directory
        base_dir = self.config["core"]["to_delete_base_dir"]
        
        # Replace variables
        base_dir = base_dir.replace("{project_dir}", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # Create category-specific directory
        to_delete_dir = os.path.join(base_dir, category)
        
        # Ensure it exists
        os.makedirs(to_delete_dir, exist_ok=True)
        
        return to_delete_dir


# Singleton instance
_config_manager = None

def get_config_manager(config_file: Optional[str] = None) -> ConfigManager:
    """
    Get the singleton config manager instance.
    
    Args:
        config_file: Path to the configuration file (optional)
        
    Returns:
        ConfigManager instance
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_file)
    return _config_manager
