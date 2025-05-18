#!/usr/bin/env python3
"""
SmartFileManager - Intelligent File Organization System

Main entry point for the SmartFileManager application.
This script initializes the necessary components and launches
the requested interface (CLI or future GUI).
"""

import os
import sys
import logging
import argparse
from pathlib import Path

# Ensure the package is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Core modules
from config.config import get_config_manager
from utils.llm_handler import get_llm_handler
from ui.cli import SmartFileManagerCLI

# Setup logging
def setup_logging(log_level="INFO"):
    """Setup logging configuration."""
    
    # Create logs directory if it doesn't exist
    logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Parse log level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    
    # Configure logging
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(logs_dir, "smartfilemanager.log"), 
                                mode='a',
                                encoding='utf-8')
        ]
    )
    
    # Return the logger
    return logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SmartFileManager - Intelligent File Organization System"
    )
    
    # General options
    parser.add_argument("--config", "-c", help="Path to custom configuration file")
    parser.add_argument("--log-level", "-l", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        default="INFO",
                        help="Set logging level")
    
    # UI options
    parser.add_argument("--gui", action="store_true", help="Launch GUI (not implemented yet)")
    
    return parser.parse_args()

def main():
    """Main entry point."""
    # Parse arguments
    args = parse_args()
    
    # Set up logging
    logger = setup_logging(args.log_level)
    logger.info("Starting SmartFileManager")
    
    try:
        # Initialize configuration
        config_manager = get_config_manager(args.config)
        logger.info("Configuration initialized")
        
        # Initialize the LLM handler if available
        llm_config = config_manager.get_config("llm")
        llm_handler = get_llm_handler(llm_config)
        
        if llm_handler.is_available():
            logger.info(f"LLM initialized with provider: {llm_handler.provider}, model: {llm_handler.model}")
        else:
            logger.warning("No LLM provider available")
        
        # Launch the appropriate interface
        if args.gui:
            logger.info("GUI mode requested but not implemented yet, falling back to CLI")
            cli = SmartFileManagerCLI()
            cli.run()
        else:
            # Launch CLI
            logger.info("Launching CLI interface")
            cli = SmartFileManagerCLI()
            cli.run()
            
    except Exception as e:
        logger.error(f"Error in SmartFileManager: {e}", exc_info=True)
        sys.exit(1)
    
    logger.info("SmartFileManager exiting")

if __name__ == "__main__":
    main()
