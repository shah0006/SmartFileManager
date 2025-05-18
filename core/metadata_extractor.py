"""
Core metadata extraction module for SmartFileManager.

This module provides a framework for extracting metadata from various file types,
with a focus on using local LLM capabilities for content-based extraction.
"""

import os
import logging
import re
from typing import Dict, List, Any, Optional, Union, Tuple
from abc import ABC, abstractmethod
from datetime import datetime

# Setup logging
logger = logging.getLogger(__name__)

class MetadataExtractor(ABC):
    """Base abstract class for all metadata extractors."""
    
    def __init__(self):
        """Initialize the metadata extractor."""
        self.file_path = None
        self.metadata = {}
    
    @abstractmethod
    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary of metadata
        """
        pass
    
    def get_file_modification_date(self, file_path: str) -> str:
        """
        Get file modification date as fallback metadata.
        
        Args:
            file_path: Path to the file
            
        Returns:
            File modification date as string
        """
        try:
            mod_time = os.path.getmtime(file_path)
            date_str = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d')
            return date_str
        except Exception as e:
            logger.error(f"Error getting modification date for {file_path}: {e}")
            return "Unknown"
    
    def extract_year_from_string(self, text: str) -> Optional[str]:
        """
        Extract a year from text (typically 1900-2099 range).
        
        Args:
            text: Text to search for year
            
        Returns:
            Extracted year or None
        """
        # Look for years in a reasonable range (19xx to 20xx)
        year_match = re.search(r'\b(19\d{2}|20\d{2})\b', text)
        if year_match:
            return year_match.group(1)
        return None
    
    def extract_from_filename(self, filename: str) -> Dict[str, Any]:
        """
        Extract basic metadata from filename as fallback.
        
        Args:
            filename: Filename to parse
            
        Returns:
            Dictionary with basic metadata
        """
        metadata = {}
        
        # Remove extension
        base_name = os.path.splitext(filename)[0]
        
        # Try to extract year
        year = self.extract_year_from_string(base_name)
        if year:
            metadata['year'] = year
        
        # Try to split by common separators
        parts = re.split(r'[_\-\.\s]+', base_name)
        if len(parts) > 1:
            # If there are multiple parts, assume first is author/source
            metadata['author'] = parts[0]
            
            # Rest could be title
            metadata['title'] = ' '.join(parts[1:])
        else:
            # Just use the whole name as title
            metadata['title'] = base_name
        
        return metadata


class LLMMetadataExtractor(MetadataExtractor):
    """Metadata extractor using local LLM capabilities."""
    
    def __init__(self, llm_handler=None):
        """
        Initialize LLM-based metadata extractor.
        
        Args:
            llm_handler: Handler for local LLM interactions
        """
        super().__init__()
        self.llm_handler = llm_handler
    
    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata using local LLM.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary of metadata
        """
        self.file_path = file_path
        self.metadata = {}
        
        # Extract basic file info
        file_name = os.path.basename(file_path)
        self.metadata['filename'] = file_name
        self.metadata['extension'] = os.path.splitext(file_name)[1].lower()
        self.metadata['file_size'] = os.path.getsize(file_path)
        self.metadata['date_modified'] = self.get_file_modification_date(file_path)
        
        # Try to extract content-based metadata with LLM if handler is available
        if self.llm_handler:
            logger.info(f"Extracting metadata from content for {file_path}")
            
            try:
                # Get text content from file - this would be implemented by subclasses
                content = self.get_file_content(file_path)
                
                if content:
                    # Use LLM to extract metadata
                    llm_metadata = self.extract_metadata_with_llm(content)
                    
                    # Merge with basic metadata
                    if llm_metadata:
                        self.metadata.update(llm_metadata)
                else:
                    logger.warning(f"Could not extract content from {file_path}")
            except Exception as e:
                logger.error(f"Error extracting metadata with LLM for {file_path}: {e}")
        
        # Fallback to filename-based extraction if key fields missing
        if not all(k in self.metadata for k in ['title', 'author']):
            logger.info(f"Using filename-based extraction as fallback for {file_path}")
            filename_metadata = self.extract_from_filename(file_name)
            
            # Only add missing fields from filename extraction
            for key, value in filename_metadata.items():
                if key not in self.metadata or not self.metadata[key]:
                    self.metadata[key] = value
        
        return self.metadata
    
    def get_file_content(self, file_path: str) -> Optional[str]:
        """
        Get text content from file for LLM processing.
        Abstract method to be implemented by subclasses.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Text content of the file or None
        """
        # This is a placeholder - subclasses should implement this
        # based on file type (PDF, EPUB, etc.)
        logger.warning("get_file_content() not implemented for base LLMMetadataExtractor")
        return None
    
    def extract_metadata_with_llm(self, content: str) -> Dict[str, Any]:
        """
        Use LLM to extract metadata from text content.
        
        Args:
            content: Text content from file
            
        Returns:
            Dictionary of metadata
        """
        if not self.llm_handler:
            logger.warning("No LLM handler available for metadata extraction")
            return {}
        
        try:
            # Prepare prompt for the LLM
            prompt = self._prepare_metadata_prompt(content)
            
            # Get LLM response
            response = self.llm_handler.get_response(prompt)
            
            # Parse the response into metadata
            parsed_metadata = self._parse_llm_response(response)
            
            logger.info(f"Extracted metadata with LLM: {parsed_metadata}")
            return parsed_metadata
            
        except Exception as e:
            logger.error(f"Error in LLM metadata extraction: {e}")
            return {}
    
    def _prepare_metadata_prompt(self, content: str) -> str:
        """
        Prepare a prompt for the LLM to extract metadata.
        
        Args:
            content: Text content from file
            
        Returns:
            Formatted prompt string
        """
        # Truncate content if it's very long
        max_length = 5000  # Adjust based on your LLM's capabilities
        truncated_content = content[:max_length] if len(content) > max_length else content
        
        prompt = f"""
        Extract the following metadata from this document text:
        1. Title
        2. Author(s)
        3. Year of publication
        4. Publisher (if present)
        5. Keywords or topics (if identifiable)

        Please format your response as:
        TITLE: [title]
        AUTHOR: [author]
        YEAR: [year]
        PUBLISHER: [publisher]
        KEYWORDS: [keyword1, keyword2, ...]

        Here is the document text:
        {truncated_content}
        """
        
        return prompt
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the LLM response into structured metadata.
        
        Args:
            response: Raw LLM response text
            
        Returns:
            Dictionary of metadata
        """
        metadata = {}
        
        # Extract fields using regex
        title_match = re.search(r'TITLE:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if title_match and title_match.group(1).strip():
            metadata['title'] = title_match.group(1).strip()
        
        author_match = re.search(r'AUTHOR(?:S)?:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if author_match and author_match.group(1).strip():
            author_text = author_match.group(1).strip()
            # Split multiple authors if present
            authors = [a.strip() for a in re.split(r'[,;&]', author_text) if a.strip()]
            if len(authors) == 1:
                metadata['author'] = authors[0]
            else:
                metadata['authors'] = authors
                metadata['author'] = authors[0]  # Primary author
        
        year_match = re.search(r'YEAR:\s*(\d{4})', response, re.IGNORECASE)
        if year_match:
            metadata['year'] = year_match.group(1)
        
        publisher_match = re.search(r'PUBLISHER:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if publisher_match and publisher_match.group(1).strip() not in ["Unknown", "N/A"]:
            metadata['publisher'] = publisher_match.group(1).strip()
        
        keywords_match = re.search(r'KEYWORDS?:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if keywords_match and keywords_match.group(1).strip():
            keywords_text = keywords_match.group(1).strip()
            keywords = [k.strip() for k in re.split(r'[,;]', keywords_text) if k.strip()]
            if keywords:
                metadata['keywords'] = keywords
        
        return metadata
