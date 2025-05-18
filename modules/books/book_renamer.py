"""
Book Renaming Module.

This module provides functionality for renaming book files (EPUB, MOBI, etc.)
based on extracted metadata using the book metadata extractor.
"""

import os
import re
import logging
import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from core.file_operations import FileOperations
from modules.books.book_metadata import BookMetadataExtractor

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    from core.file_operations import tqdm

# Setup logging
logger = logging.getLogger(__name__)

class BookRenamer:
    """Book renaming functionality."""
    
    def __init__(self, llm_handler=None):
        """
        Initialize the book renamer.
        
        Args:
            llm_handler: Handler for local LLM interactions
        """
        self.llm_handler = llm_handler
        self.file_ops = FileOperations()
        self.metadata_extractor = BookMetadataExtractor(llm_handler=llm_handler)
        
        # Supported book extensions
        self.book_extensions = [".epub", ".mobi", ".azw", ".azw3", ".pdf", ".txt"]
    
    def process_directory(self, 
                         directory_path: str,
                         recursive: bool = True,
                         rename_immediately: bool = True) -> Tuple[int, int, int]:
        """
        Process a directory of book files for renaming.
        
        Args:
            directory_path: Path to directory containing book files
            recursive: Whether to search subdirectories
            rename_immediately: Whether to rename files immediately (True) or just return operations (False)
            
        Returns:
            Tuple of (total_files, renamed_files, error_count)
        """
        # Scan for book files
        book_files = self.file_ops.scan_directory(
            directory_path,
            file_types=self.book_extensions,
            recursive=recursive
        )
        
        logger.info(f"Found {len(book_files)} book files in {directory_path}")
        
        # Process files in batches with progress tracking
        rename_operations = []
        error_count = 0
        success_count = 0
        
        # Process with progress tracking
        for file_path in tqdm(book_files, desc="Analyzing books"):
            try:
                operation = self.process_file(file_path)
                if operation:
                    rename_operations.append(operation)
                    
                    # If immediate renaming is enabled, rename the file now
                    if rename_immediately and operation["should_rename"]:
                        result = self.execute_rename(operation)
                        if result:
                            success_count += 1
                        else:
                            error_count += 1
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                error_count += 1
        
        if not rename_immediately:
            logger.info(f"Generated {len(rename_operations)} rename operations")
        else:
            logger.info(f"Renamed {success_count} files, encountered {error_count} errors")
        
        return len(book_files), success_count, error_count
    
    def process_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Process a single book file to extract metadata and determine new filename.
        
        Args:
            file_path: Path to the book file
            
        Returns:
            Dictionary with rename operation details or None if no rename needed
        """
        file_path = os.path.abspath(file_path)
        original_filename = os.path.basename(file_path)
        
        # Check if already properly named
        if self._is_already_properly_named(original_filename):
            logger.info(f"File is already properly named: {original_filename}")
            return None
        
        # Extract metadata
        try:
            metadata = self.metadata_extractor.extract_metadata(file_path)
            
            # Generate new filename
            new_filename = self.metadata_extractor.generate_filename(metadata)
            
            # Check if rename is needed
            should_rename = original_filename != new_filename
            
            # Return operation details
            operation = {
                "original_path": file_path,
                "original_name": original_filename,
                "new_name": new_filename,
                "new_path": os.path.join(os.path.dirname(file_path), new_filename),
                "metadata": metadata,
                "should_rename": should_rename
            }
            
            return operation
        except Exception as e:
            logger.error(f"Error processing book file {file_path}: {e}")
            return None
    
    def _is_already_properly_named(self, filename: str) -> bool:
        """
        Check if a filename is already properly formatted.
        
        Args:
            filename: Filename to check
            
        Returns:
            True if properly named, False otherwise
        """
        # Pattern: Author_Year_Title.ext or Author_Title.ext
        pattern = r'^[^_]+_(\d{4}_)?[^_]+\.[a-zA-Z0-9]+$'
        return bool(re.match(pattern, filename))
    
    def execute_rename(self, operation: Dict[str, Any]) -> bool:
        """
        Execute a rename operation.
        
        Args:
            operation: Dictionary with rename operation details
            
        Returns:
            Success status
        """
        if not operation["should_rename"]:
            logger.info(f"Skipping rename for {operation['original_name']} (already correct)")
            return True
        
        # Perform the rename using our safe file operations
        result = self.file_ops.safe_move(
            operation["original_path"],
            operation["new_path"],
            category="books",
            overwrite=False
        )
        
        if result:
            logger.info(f"Renamed: {operation['original_name']} -> {operation['new_name']}")
        else:
            logger.warning(f"Failed to rename: {operation['original_name']}")
        
        return result
    
    def generate_rename_report(self, operations: List[Dict[str, Any]]) -> str:
        """
        Generate a detailed rename operations report.
        
        Args:
            operations: List of rename operation dictionaries
            
        Returns:
            Path to the generated report file
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(os.getcwd(), f"book_rename_report_{timestamp}.txt")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("Book Rename Operations Report\n")
            f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total files processed: {len(operations)}\n")
            f.write(f"Files to rename: {sum(1 for op in operations if op['should_rename'])}\n\n")
            f.write("Detailed rename operations:\n")
            
            for i, op in enumerate(operations, 1):
                if op["should_rename"]:
                    f.write(f"\n{i}. Original: {op['original_name']}\n")
                    f.write(f"   New name: {op['new_name']}\n")
                    
                    if op['metadata'] and op['metadata'].get('title'):
                        f.write(f"   Title: {op['metadata'].get('title')}\n")
                    if op['metadata'] and op['metadata'].get('author'):
                        f.write(f"   Author: {op['metadata'].get('author')}\n")
                    if op['metadata'] and op['metadata'].get('year'):
                        f.write(f"   Year: {op['metadata'].get('year')}\n")
                    if op['metadata'] and op['metadata'].get('isbn'):
                        f.write(f"   ISBN: {op['metadata'].get('isbn')}\n")
        
        logger.info(f"Created detailed report: {report_file}")
        return report_file


class SmartBookRenamer(BookRenamer):
    """Enhanced book renamer with custom naming templates."""
    
    def __init__(self, llm_handler=None, naming_template=None):
        """
        Initialize the smart book renamer.
        
        Args:
            llm_handler: Handler for local LLM interactions
            naming_template: Optional custom naming template
        """
        super().__init__(llm_handler=llm_handler)
        
        # Set naming template
        self.naming_template = naming_template or "{author}_{year}_{title}"
    
    def generate_filename(self, metadata: Dict[str, Any]) -> str:
        """
        Generate a filename using a custom template.
        
        Args:
            metadata: Dictionary of metadata
            
        Returns:
            Generated filename
        """
        # Get clean values for template
        clean_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, str):
                clean_metadata[key] = self.metadata_extractor._clean_for_filename(value)
            else:
                clean_metadata[key] = value
        
        # Add some defaults if missing
        if "author" not in clean_metadata:
            clean_metadata["author"] = "Unknown"
        if "title" not in clean_metadata:
            clean_metadata["title"] = "Untitled"
        if "year" not in clean_metadata:
            clean_metadata["year"] = ""
        
        # Truncate title if too long
        if len(clean_metadata.get("title", "")) > 50:
            clean_metadata["title"] = clean_metadata["title"][:47] + "..."
        
        # Get the original extension
        ext = metadata.get("extension", ".epub")
        
        try:
            # Format using template
            basename = self.naming_template.format(**clean_metadata)
            # Clean up any remaining invalid characters
            basename = re.sub(r'[^\w\-_]', '_', basename)
            # Remove consecutive underscores
            basename = re.sub(r'_+', '_', basename)
            # Ensure filename doesn't end with underscore before extension
            basename = basename.rstrip('_')
            
            return f"{basename}{ext}"
        except KeyError as e:
            logger.error(f"Invalid key in naming template: {e}")
            # Fall back to default format
            if clean_metadata.get("year"):
                return f"{clean_metadata['author']}_{clean_metadata['year']}_{clean_metadata['title']}{ext}"
            else:
                return f"{clean_metadata['author']}_{clean_metadata['title']}{ext}"
        except Exception as e:
            logger.error(f"Error formatting filename: {e}")
            return f"{clean_metadata['author']}_{clean_metadata['title']}{ext}"
