"""
Core file operations module for SmartFileManager.

This module provides safe file operations with progress tracking
and "to-delete" functionality instead of permanent deletion.
"""

import os
import shutil
import logging
import datetime
from pathlib import Path
from typing import List, Dict, Union, Optional, Callable

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    from tqdm.auto import tqdm as auto_tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Simple fallback progress function
    def tqdm(iterable, **kwargs):
        desc = kwargs.get('desc', 'Processing')
        total = kwargs.get('total', None)
        if total is None:
            try:
                total = len(iterable)
            except Exception:
                total = '?'
        print(f"{desc} - Started (0/{total})")
        count = 0
        for item in iterable:
            count += 1
            if count % 10 == 0 or count == 1:
                print(f"{desc} - Progress ({count}/{total})")
            yield item
        print(f"{desc} - Completed ({count}/{total})")
    auto_tqdm = tqdm  # Alias for consistency

# Setup logging
logger = logging.getLogger(__name__)

class FileOperations:
    """Core file operations with safety measures and progress tracking."""
    
    def __init__(self, root_dir: str = None):
        """
        Initialize file operations manager.
        
        Args:
            root_dir: Root directory for operations (optional)
        """
        self.root_dir = root_dir
        # Create a dictionary to store to-delete folders by category
        self.to_delete_folders = {}
        
    def get_to_delete_folder(self, category: str) -> str:
        """
        Get or create a to-delete folder for the specified category.
        
        Args:
            category: Category of files (e.g., 'books', 'academic', 'medical')
            
        Returns:
            Path to the to-delete folder
        """
        if category not in self.to_delete_folders:
            if self.root_dir:
                base_dir = self.root_dir
            else:
                base_dir = os.path.expanduser("~")
            
            # Create category-specific to-delete folder
            to_delete_dir = os.path.join(base_dir, f"To_Delete_{category}")
            os.makedirs(to_delete_dir, exist_ok=True)
            self.to_delete_folders[category] = to_delete_dir
            logger.info(f"Created to-delete folder for {category}: {to_delete_dir}")
            
        return self.to_delete_folders[category]
    
    def safe_move(self, 
                 source_path: str, 
                 dest_path: str, 
                 category: str = "general",
                 overwrite: bool = False) -> bool:
        """
        Safely move a file, with error handling.
        
        Args:
            source_path: Source file path
            dest_path: Destination file path
            category: File category (for to-delete folder)
            overwrite: Whether to overwrite existing files
            
        Returns:
            Success status
        """
        try:
            source_path = os.path.abspath(source_path)
            dest_path = os.path.abspath(dest_path)
            
            # Check for circular moves
            if os.path.dirname(source_path) == os.path.dirname(dest_path):
                if os.path.basename(source_path).lower() == os.path.basename(dest_path).lower():
                    logger.warning(f"Skipping circular move: {source_path} -> {dest_path}")
                    return False
            
            # Ensure source exists
            if not os.path.exists(source_path):
                logger.error(f"Source file does not exist: {source_path}")
                return False
            
            # Handle existing destination
            if os.path.exists(dest_path):
                if overwrite:
                    # Move to to-delete folder instead of deleting
                    to_delete_dir = self.get_to_delete_folder(category)
                    delete_filename = f"{os.path.basename(dest_path)}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
                    delete_path = os.path.join(to_delete_dir, delete_filename)
                    
                    try:
                        shutil.move(dest_path, delete_path)
                        logger.info(f"Moved existing file to to-delete folder: {dest_path} -> {delete_path}")
                    except Exception as e:
                        logger.error(f"Error moving existing file to to-delete folder: {e}")
                        return False
                else:
                    logger.warning(f"Destination file already exists: {dest_path}")
                    return False
            
            # Ensure destination directory exists
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            # Perform the move
            shutil.move(source_path, dest_path)
            logger.info(f"Successfully moved: {source_path} -> {dest_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error moving file {source_path} to {dest_path}: {e}")
            return False
    
    def safe_copy(self, 
                 source_path: str, 
                 dest_path: str,
                 overwrite: bool = False) -> bool:
        """
        Safely copy a file, with error handling.
        
        Args:
            source_path: Source file path
            dest_path: Destination file path
            overwrite: Whether to overwrite existing files
            
        Returns:
            Success status
        """
        try:
            source_path = os.path.abspath(source_path)
            dest_path = os.path.abspath(dest_path)
            
            # Ensure source exists
            if not os.path.exists(source_path):
                logger.error(f"Source file does not exist: {source_path}")
                return False
            
            # Handle existing destination
            if os.path.exists(dest_path) and not overwrite:
                logger.warning(f"Destination file already exists: {dest_path}")
                return False
            
            # Ensure destination directory exists
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            # Perform the copy
            shutil.copy2(source_path, dest_path)
            logger.info(f"Successfully copied: {source_path} -> {dest_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error copying file {source_path} to {dest_path}: {e}")
            return False
    
    def safe_delete(self, file_path: str, category: str = "general") -> bool:
        """
        Safely 'delete' a file by moving it to a to-delete folder.
        
        Args:
            file_path: Path to file to delete
            category: File category (for to-delete folder)
            
        Returns:
            Success status
        """
        try:
            file_path = os.path.abspath(file_path)
            
            # Ensure file exists
            if not os.path.exists(file_path):
                logger.error(f"File does not exist: {file_path}")
                return False
            
            # Get to-delete folder for this category
            to_delete_dir = self.get_to_delete_folder(category)
            
            # Create a unique filename with timestamp
            delete_filename = f"{os.path.basename(file_path)}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            delete_path = os.path.join(to_delete_dir, delete_filename)
            
            # Move to to-delete folder
            shutil.move(file_path, delete_path)
            logger.info(f"Moved file to to-delete folder: {file_path} -> {delete_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error moving file to to-delete folder: {e}")
            return False
    
    def batch_process_files(self, 
                           file_list: List[str], 
                           operation: Callable,
                           desc: str = "Processing files",
                           **kwargs) -> Dict[str, int]:
        """
        Process multiple files with progress tracking.
        
        Args:
            file_list: List of file paths to process
            operation: Function to call for each file
            desc: Description for progress bar
            **kwargs: Additional arguments to pass to operation
            
        Returns:
            Statistics about the operation
        """
        success_count = 0
        error_count = 0
        skipped_count = 0
        
        # Use tqdm for progress tracking if available
        for file_path in tqdm(file_list, desc=desc):
            try:
                result = operation(file_path, **kwargs)
                if result is True:
                    success_count += 1
                elif result is False:
                    skipped_count += 1
                else:
                    # For operations that return something other than boolean
                    success_count += 1
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                error_count += 1
        
        stats = {
            'total': len(file_list),
            'success': success_count,
            'error': error_count,
            'skipped': skipped_count
        }
        
        logger.info(f"Batch processing complete. Stats: {stats}")
        return stats
    
    def scan_directory(self, 
                      directory: str, 
                      file_types: List[str] = None,
                      recursive: bool = True,
                      exclude_dirs: List[str] = None) -> List[str]:
        """
        Scan a directory for files with optional filtering.
        
        Args:
            directory: Directory to scan
            file_types: List of file extensions to include (e.g., ['.pdf', '.epub'])
            recursive: Whether to search subdirectories
            exclude_dirs: Directories to exclude from search
            
        Returns:
            List of file paths
        """
        file_list = []
        
        if exclude_dirs is None:
            exclude_dirs = []
        
        exclude_dirs = [os.path.normpath(d.lower()) for d in exclude_dirs]
        
        try:
            for root, dirs, files in os.walk(directory):
                # Skip excluded directories
                dirs[:] = [d for d in dirs if os.path.normpath(os.path.join(root, d).lower()) not in exclude_dirs]
                
                # If not recursive, clear dirs to prevent further walking
                if not recursive:
                    dirs[:] = []
                
                for filename in files:
                    # Filter by file type if specified
                    if file_types is not None:
                        if not any(filename.lower().endswith(ext.lower()) for ext in file_types):
                            continue
                    
                    file_path = os.path.join(root, filename)
                    file_list.append(file_path)
            
            logger.info(f"Found {len(file_list)} files in {directory}")
            return file_list
            
        except Exception as e:
            logger.error(f"Error scanning directory {directory}: {e}")
            return []
