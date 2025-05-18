"""
PDF Metadata Extraction Module for Academic Papers.

This module extends the core metadata extraction framework
to handle PDF files specifically for academic papers.
"""

import os
import re
import logging
from typing import Dict, List, Any, Optional, Tuple

from core.metadata_extractor import LLMMetadataExtractor

# Setup logging
logger = logging.getLogger(__name__)

class PDFMetadataExtractor(LLMMetadataExtractor):
    """Metadata extractor specialized for academic PDF files."""
    
    def __init__(self, llm_handler=None):
        """
        Initialize the PDF metadata extractor.
        
        Args:
            llm_handler: Handler for local LLM interactions
        """
        super().__init__(llm_handler=llm_handler)
    
    def get_file_content(self, file_path: str) -> Optional[str]:
        """
        Extract text content from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text content or None if extraction fails
        """
        try:
            # First try PyMuPDF (fitz) as it's faster
            return self._extract_with_pymupdf(file_path)
        except ImportError:
            logger.warning("PyMuPDF not available, trying pdfplumber")
            try:
                return self._extract_with_pdfplumber(file_path)
            except ImportError:
                logger.warning("pdfplumber not available, trying pdfminer")
                try:
                    return self._extract_with_pdfminer(file_path)
                except ImportError:
                    logger.error("No PDF extraction libraries available")
                    return None
        except Exception as e:
            logger.error(f"Error extracting PDF content: {e}")
            return None
    
    def _extract_with_pymupdf(self, file_path: str) -> str:
        """Extract PDF text using PyMuPDF (fitz)."""
        import fitz  # PyMuPDF

        text_content = ""
        try:
            doc = fitz.open(file_path)
            
            # Extract text from the first few pages (title, abstract, etc.)
            max_pages = min(10, doc.page_count)
            for page_num in range(max_pages):
                page = doc[page_num]
                text_content += page.get_text()
            
            doc.close()
            logger.info(f"Extracted {max_pages} pages with PyMuPDF")
            return text_content
            
        except Exception as e:
            logger.error(f"Error extracting with PyMuPDF: {e}")
            raise
    
    def _extract_with_pdfplumber(self, file_path: str) -> str:
        """Extract PDF text using pdfplumber."""
        import pdfplumber

        text_content = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                # Extract text from the first few pages
                max_pages = min(10, len(pdf.pages))
                for page_num in range(max_pages):
                    page = pdf.pages[page_num]
                    text_content += page.extract_text() or ""
            
            logger.info(f"Extracted {max_pages} pages with pdfplumber")
            return text_content
            
        except Exception as e:
            logger.error(f"Error extracting with pdfplumber: {e}")
            raise
    
    def _extract_with_pdfminer(self, file_path: str) -> str:
        """Extract PDF text using pdfminer."""
        from pdfminer.high_level import extract_text

        try:
            text_content = extract_text(file_path, page_numbers=list(range(10)))
            logger.info("Extracted text with pdfminer")
            return text_content
            
        except Exception as e:
            logger.error(f"Error extracting with pdfminer: {e}")
            raise
    
    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from an academic PDF.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary of metadata
        """
        # First call the parent class to extract basic metadata
        metadata = super().extract_metadata(file_path)
        
        # Then add PDF-specific metadata extraction
        try:
            # Try to extract PDF document info using PyMuPDF
            pdf_info = self._extract_pdf_document_info(file_path)
            if pdf_info:
                # Only add fields that aren't already extracted by the LLM
                for key, value in pdf_info.items():
                    if key not in metadata or not metadata[key]:
                        metadata[key] = value
        except Exception as e:
            logger.error(f"Error extracting PDF document info: {e}")
        
        # Clean and normalize metadata
        metadata = self._clean_metadata(metadata)
        
        return metadata
    
    def _extract_pdf_document_info(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from PDF document info dictionary.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dictionary of metadata from PDF document info
        """
        metadata = {}
        
        try:
            import fitz  # PyMuPDF
            
            doc = fitz.open(file_path)
            
            # Get document info dictionary
            info = doc.metadata
            if info:
                # Map PDF metadata to our format
                if info.get("title"):
                    metadata["title"] = info["title"]
                
                if info.get("author"):
                    # Handle multiple authors
                    authors = info["author"]
                    if ";" in authors:
                        author_list = [a.strip() for a in authors.split(";")]
                        metadata["authors"] = author_list
                        metadata["author"] = author_list[0]
                    elif "," in authors:
                        author_list = [a.strip() for a in authors.split(",")]
                        metadata["authors"] = author_list
                        metadata["author"] = author_list[0]
                    else:
                        metadata["author"] = authors
                
                if info.get("subject"):
                    metadata["subject"] = info["subject"]
                
                if info.get("keywords"):
                    metadata["keywords"] = info["keywords"]
                
                # Try to extract year from various fields
                if info.get("creationDate"):
                    date_match = re.search(r'(\d{4})', info["creationDate"])
                    if date_match:
                        metadata["year"] = date_match.group(1)
                
                if "year" not in metadata and info.get("modDate"):
                    date_match = re.search(r'(\d{4})', info["modDate"])
                    if date_match:
                        metadata["year"] = date_match.group(1)
            
            doc.close()
            return metadata
            
        except ImportError:
            logger.warning("PyMuPDF not available for document info extraction")
            return metadata
        except Exception as e:
            logger.error(f"Error extracting PDF document info: {e}")
            return metadata
    
    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean and normalize metadata.
        
        Args:
            metadata: Raw metadata dictionary
            
        Returns:
            Cleaned metadata dictionary
        """
        # Make a copy to avoid modifying the original
        cleaned = metadata.copy()
        
        # Clean title
        if "title" in cleaned and cleaned["title"]:
            title = cleaned["title"]
            # Remove line breaks and extra spaces
            title = re.sub(r'\s+', ' ', title).strip()
            # Remove common prefixes
            title = re.sub(r'^(Title|TITLE|Title:)\s*', '', title)
            cleaned["title"] = title
        
        # Clean author
        if "author" in cleaned and cleaned["author"]:
            author = cleaned["author"]
            # Remove line breaks and extra spaces
            author = re.sub(r'\s+', ' ', author).strip()
            # Remove common prefixes
            author = re.sub(r'^(Author|AUTHORS|Author:)\s*', '', author)
            cleaned["author"] = author
        
        # Clean year
        if "year" in cleaned and cleaned["year"]:
            year = cleaned["year"]
            # Extract just the 4-digit year
            year_match = re.search(r'(19\d{2}|20\d{2})', str(year))
            if year_match:
                cleaned["year"] = year_match.group(1)
        
        return cleaned
    
    def generate_filename(self, metadata: Dict[str, Any]) -> str:
        """
        Generate a standardized filename from metadata.
        
        Args:
            metadata: Dictionary of metadata
            
        Returns:
            Standardized filename
        """
        # Start with default components
        author = metadata.get("author", "Unknown")
        year = metadata.get("year", "")
        title = metadata.get("title", "Untitled")
        
        # Clean components for filename use
        author = self._clean_for_filename(author)
        title = self._clean_for_filename(title)
        
        # Truncate title if too long
        if len(title) > 50:
            title = title[:47] + "..."
        
        # Format: Author_Year_Title.pdf
        if year:
            filename = f"{author}_{year}_{title}.pdf"
        else:
            filename = f"{author}_{title}.pdf"
        
        return filename
    
    def _clean_for_filename(self, text: str) -> str:
        """
        Clean text for use in a filename.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text suitable for filenames
        """
        # Replace special characters with spaces
        text = re.sub(r'[^\w\s-]', ' ', text)
        
        # Replace spaces with underscores
        text = re.sub(r'\s+', '_', text.strip())
        
        # Remove consecutive underscores
        text = re.sub(r'_+', '_', text)
        
        return text
