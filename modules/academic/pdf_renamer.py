"""
PDF Renaming Module for Academic Papers.

This module provides functionality for renaming academic PDF files
based on extracted metadata using the PDF metadata extractor.
"""

import os
import re
import logging
import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from core.file_operations import FileOperations
from modules.academic.pdf_metadata import PDFMetadataExtractor

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    from core.file_operations import tqdm

# Setup logging
logger = logging.getLogger(__name__)

class PDFRenamer:
    """Academic PDF renaming functionality."""
    
    def __init__(self, llm_handler=None):
        """
        Initialize the PDF renamer.
        
        Args:
            llm_handler: Handler for local LLM interactions
        """
        self.llm_handler = llm_handler
        self.file_ops = FileOperations()
        self.metadata_extractor = PDFMetadataExtractor(llm_handler=llm_handler)
    
    def process_directory(self, 
                         directory_path: str,
                         recursive: bool = True,
                         rename_immediately: bool = True) -> Tuple[int, int, int]:
        """
        Process a directory of PDF files for renaming.
        
        Args:
            directory_path: Path to directory containing PDF files
            recursive: Whether to search subdirectories
            rename_immediately: Whether to rename files immediately (True) or just return operations (False)
            
        Returns:
            Tuple of (total_files, renamed_files, error_count)
        """
        # Scan for PDF files
        pdf_files = self.file_ops.scan_directory(
            directory_path,
            file_types=[".pdf"],
            recursive=recursive
        )
        
        logger.info(f"Found {len(pdf_files)} PDF files in {directory_path}")
        
        # Process files in batches with progress tracking
        rename_operations = []
        error_count = 0
        success_count = 0
        
        # Process with progress tracking
        for file_path in tqdm(pdf_files, desc="Analyzing PDFs"):
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
        
        return len(pdf_files), success_count, error_count
    
    def process_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Process a single PDF file to extract metadata and determine new filename.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary with rename operation details or None if no rename needed
        """
        file_path = os.path.abspath(file_path)
        original_filename = os.path.basename(file_path)
        
        # Extract metadata
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
            category="academic",
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
        report_file = os.path.join(os.getcwd(), f"rename_report_{timestamp}.txt")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("PDF Rename Operations Report\n")
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
        
        logger.info(f"Created detailed report: {report_file}")
        return report_file


class ZoteroStyleRenamer(PDFRenamer):
    """PDF renamer that follows Zotero naming conventions."""
    
    def __init__(self, llm_handler=None):
        """
        Initialize the Zotero-style PDF renamer.
        
        Args:
            llm_handler: Handler for local LLM interactions
        """
        super().__init__(llm_handler=llm_handler)
    
    def _clean_for_zotero_filename(self, text: str) -> str:
        """
        Clean text specifically for Zotero-compatible filenames.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text suitable for Zotero filenames
        """
        # Replace special characters with spaces
        text = re.sub(r'[^\w\s-]', ' ', text)
        
        # Replace spaces with space (Zotero uses spaces, not underscores)
        text = re.sub(r'\s+', ' ', text.strip())
        
        return text
    
    def generate_zotero_filename(self, metadata: Dict[str, Any]) -> str:
        """
        Generate a Zotero-style filename from metadata.
        
        Args:
            metadata: Dictionary of metadata
            
        Returns:
            Standardized Zotero-style filename
        """
        # Start with default components
        author = metadata.get("author", "Unknown")
        year = metadata.get("year", "")
        title = metadata.get("title", "Untitled")
        
        # Clean components for filename use
        author = self._clean_for_zotero_filename(author)
        title = self._clean_for_zotero_filename(title)
        
        # Truncate title if too long
        if len(title) > 50:
            title = title[:47] + "..."
        
        # Format: Author - Year - Title.pdf (Zotero format)
        if year:
            filename = f"{author} - {year} - {title}.pdf"
        else:
            filename = f"{author} - {title}.pdf"
        
        return filename
    
    def process_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Process a file using Zotero naming conventions.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary with rename operation details or None if no rename needed
        """
        file_path = os.path.abspath(file_path)
        original_filename = os.path.basename(file_path)
        
        # Extract metadata
        metadata = self.metadata_extractor.extract_metadata(file_path)
        
        # Generate new filename using Zotero style
        new_filename = self.generate_zotero_filename(metadata)
        
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
