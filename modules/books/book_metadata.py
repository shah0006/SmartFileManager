"""
Book Metadata Extraction Module.

This module extends the core metadata extraction framework
to handle book files specifically (EPUB, MOBI, etc.).
"""

import os
import re
import logging
from typing import Dict, Any, Optional
import zipfile
import xml.etree.ElementTree as ET

from core.metadata_extractor import LLMMetadataExtractor

# Setup logging
logger = logging.getLogger(__name__)

class BookMetadataExtractor(LLMMetadataExtractor):
    """Metadata extractor specialized for book files."""
    
    def __init__(self, llm_handler=None):
        """
        Initialize the book metadata extractor.
        
        Args:
            llm_handler: Handler for local LLM interactions
        """
        super().__init__(llm_handler=llm_handler)
    
    def get_file_content(self, file_path: str) -> Optional[str]:
        """
        Extract text content from a book file.
        
        Args:
            file_path: Path to the book file
            
        Returns:
            Extracted text content or None if extraction fails
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == ".epub":
                return self._extract_epub_content(file_path)
            elif file_ext == ".mobi" or file_ext == ".azw" or file_ext == ".azw3":
                return self._extract_mobi_content(file_path)
            elif file_ext == ".txt":
                return self._extract_text_content(file_path)
            elif file_ext == ".pdf":
                return self._extract_pdf_content(file_path)
            else:
                logger.warning(f"Unsupported file format for content extraction: {file_ext}")
                return None
        except Exception as e:
            logger.error(f"Error extracting content from {file_path}: {e}")
            return None
    
    def _extract_epub_content(self, file_path: str) -> str:
        """Extract text content from EPUB file."""
        try:
            text_content = ""
            
            with zipfile.ZipFile(file_path, "r") as epub:
                # Find the content files
                content_files = [
                    name for name in epub.namelist()
                    if name.endswith(".html") or name.endswith(".xhtml") or name.endswith(".htm")
                ]
                
                # Sort by filename to maintain order
                content_files.sort()
                
                # Take only the first few content files (title, intro, etc.)
                sample_files = content_files[:5] if len(content_files) > 5 else content_files
                
                for content_file in sample_files:
                    try:
                        with epub.open(content_file) as f:
                            html_content = f.read().decode("utf-8")
                            
                            # Simple HTML to text conversion
                            text = re.sub(r"<[^>]+>", " ", html_content)
                            text = re.sub(r"\s+", " ", text).strip()
                            
                            text_content += text + "\n\n"
                    except Exception as e:
                        logger.error(f"Error extracting content from {content_file}: {e}")
                
                # Extract metadata from content.opf
                opf_files = [name for name in epub.namelist() if name.endswith(".opf")]
                if opf_files:
                    try:
                        with epub.open(opf_files[0]) as f:
                            opf_content = f.read().decode("utf-8")
                            
                            # Add metadata as header to the text content
                            metadata_text = self._parse_opf_metadata(opf_content)
                            if metadata_text:
                                text_content = metadata_text + "\n\n" + text_content
                    except Exception as e:
                        logger.error(f"Error extracting metadata from OPF: {e}")
            
            return text_content
        except Exception as e:
            logger.error(f"Error extracting EPUB content: {e}")
            raise
    
    def _parse_opf_metadata(self, opf_content: str) -> str:
        """Parse metadata from OPF content."""
        metadata_text = ""
        
        try:
            # Parse XML
            namespace = {"dc": "http://purl.org/dc/elements/1.1/", "opf": "http://www.idpf.org/2007/opf"}
            root = ET.fromstring(opf_content)
            
            # Find metadata element
            metadata_elem = root.find(".//metadata") or root.find(".//{http://www.idpf.org/2007/opf}metadata")
            
            if metadata_elem is not None:
                # Extract title
                title_elem = metadata_elem.find(".//dc:title", namespace) or metadata_elem.find(".//{http://purl.org/dc/elements/1.1/}title")
                if title_elem is not None and title_elem.text:
                    metadata_text += f"Title: {title_elem.text}\n"
                
                # Extract creator/author
                creator_elems = metadata_elem.findall(".//dc:creator", namespace) or metadata_elem.findall(".//{http://purl.org/dc/elements/1.1/}creator")
                if creator_elems:
                    authors = [elem.text for elem in creator_elems if elem.text]
                    if authors:
                        metadata_text += f"Author: {', '.join(authors)}\n"
                
                # Extract published date
                date_elem = metadata_elem.find(".//dc:date", namespace) or metadata_elem.find(".//{http://purl.org/dc/elements/1.1/}date")
                if date_elem is not None and date_elem.text:
                    date_text = date_elem.text
                    # Try to extract year
                    year_match = re.search(r"(\d{4})", date_text)
                    if year_match:
                        metadata_text += f"Year: {year_match.group(1)}\n"
                    else:
                        metadata_text += f"Date: {date_text}\n"
                
                # Extract publisher
                publisher_elem = metadata_elem.find(".//dc:publisher", namespace) or metadata_elem.find(".//{http://purl.org/dc/elements/1.1/}publisher")
                if publisher_elem is not None and publisher_elem.text:
                    metadata_text += f"Publisher: {publisher_elem.text}\n"
                
                # Extract language
                language_elem = metadata_elem.find(".//dc:language", namespace) or metadata_elem.find(".//{http://purl.org/dc/elements/1.1/}language")
                if language_elem is not None and language_elem.text:
                    metadata_text += f"Language: {language_elem.text}\n"
                
                # Extract ISBN
                identifier_elems = metadata_elem.findall(".//dc:identifier", namespace) or metadata_elem.findall(".//{http://purl.org/dc/elements/1.1/}identifier")
                if identifier_elems:
                    for elem in identifier_elems:
                        id_text = elem.text
                        if id_text and ("isbn" in id_text.lower() or re.search(r"\d{10,13}", id_text)):
                            # Extract just the digits for ISBN
                            isbn_match = re.search(r"(?:isbn[:\s]*)?(\d[\d\-]+\d)", id_text, re.IGNORECASE)
                            if isbn_match:
                                metadata_text += f"ISBN: {isbn_match.group(1)}\n"
                                break
        
        except Exception as e:
            logger.error(f"Error parsing OPF metadata: {e}")
        
        return metadata_text
    
    def _extract_mobi_content(self, file_path: str) -> str:
        """Extract text content from MOBI file."""
        try:
            # Try to use mobi-python library
            try:
                import mobi
                book = mobi.Mobi(file_path)
                content = book.text
                
                # Truncate content if it's very long
                max_length = 10000  # First N characters are usually enough for metadata
                truncated_content = content[:max_length] if len(content) > max_length else content
                
                return truncated_content
                
            except ImportError:
                logger.warning("mobi-python library not available")
                
                # Fall back to using calibre's ebook-convert
                return self._extract_with_calibre(file_path)
                
        except Exception as e:
            logger.error(f"Error extracting MOBI content: {e}")
            raise
    
    def _extract_with_calibre(self, file_path: str) -> str:
        """Extract text using Calibre's ebook-convert tool."""
        import subprocess
        import tempfile
        
        try:
            # Create a temporary file for the output
            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
                temp_output = temp_file.name
            
            # Call ebook-convert
            process = subprocess.run(
                ["ebook-convert", file_path, temp_output],
                capture_output=True,
                text=True
            )
            
            if process.returncode != 0:
                logger.error(f"Error converting with Calibre: {process.stderr}")
                return ""
            
            # Read the output file
            with open(temp_output, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Clean up
            os.unlink(temp_output)
            
            return content
            
        except Exception as e:
            logger.error(f"Error extracting with Calibre: {e}")
            return ""
    
    def _extract_text_content(self, file_path: str) -> str:
        """Extract content from plain text file."""
        try:
            # Try utf-8 first
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                return content
            except UnicodeDecodeError:
                # Fall back to system default encoding
                with open(file_path, "r") as f:
                    content = f.read()
                return content
                
        except Exception as e:
            logger.error(f"Error extracting text content: {e}")
            raise
    
    def _extract_pdf_content(self, file_path: str) -> str:
        """Extract text from PDF file."""
        try:
            # First try PyMuPDF (fitz) as it's faster
            try:
                import fitz  # PyMuPDF
                
                text_content = ""
                doc = fitz.open(file_path)
                
                # Extract text from the first few pages (title, TOC, intro, etc.)
                max_pages = min(20, doc.page_count)
                for page_num in range(max_pages):
                    page = doc[page_num]
                    text_content += page.get_text()
                
                doc.close()
                return text_content
                
            except ImportError:
                logger.warning("PyMuPDF not available, trying other methods")
                
                # Try pdfminer
                try:
                    from pdfminer.high_level import extract_text
                    return extract_text(file_path, page_numbers=list(range(20)))
                    
                except ImportError:
                    logger.warning("pdfminer not available")
                    return ""
                    
        except Exception as e:
            logger.error(f"Error extracting PDF content: {e}")
            raise
    
    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a book file.
        
        Args:
            file_path: Path to the book file
            
        Returns:
            Dictionary of metadata
        """
        # Call the parent class to extract basic metadata first
        metadata = super().extract_metadata(file_path)
        
        # Then add book-specific metadata extraction
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == ".epub":
                epub_metadata = self._extract_epub_metadata(file_path)
                # Update with epub-specific metadata
                metadata.update(epub_metadata)
            elif file_ext == ".mobi" or file_ext == ".azw" or file_ext == ".azw3":
                # Currently relying on LLM for mobi metadata
                pass
            elif file_ext == ".pdf":
                # Try to extract PDF-specific metadata
                pdf_metadata = self._extract_pdf_metadata(file_path)
                # Update with pdf-specific metadata
                metadata.update(pdf_metadata)
        except Exception as e:
            logger.error(f"Error extracting format-specific metadata: {e}")
        
        # Clean and normalize the metadata
        metadata = self._clean_metadata(metadata)
        
        return metadata
    
    def _extract_epub_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from EPUB file."""
        metadata = {}
        
        try:
            with zipfile.ZipFile(file_path, "r") as epub:
                # Find OPF files
                opf_files = [name for name in epub.namelist() if name.endswith(".opf")]
                
                if not opf_files:
                    return metadata
                
                # Read the first OPF file
                with epub.open(opf_files[0]) as f:
                    opf_content = f.read().decode("utf-8")
                
                # Parse with ElementTree
                namespace = {"dc": "http://purl.org/dc/elements/1.1/", "opf": "http://www.idpf.org/2007/opf"}
                root = ET.fromstring(opf_content)
                
                # Find metadata element
                metadata_elem = root.find(".//metadata") or root.find(".//{http://www.idpf.org/2007/opf}metadata")
                
                if metadata_elem is not None:
                    # Extract title
                    title_elem = metadata_elem.find(".//dc:title", namespace) or metadata_elem.find(".//{http://purl.org/dc/elements/1.1/}title")
                    if title_elem is not None and title_elem.text:
                        metadata["title"] = title_elem.text
                    
                    # Extract creator/author
                    creator_elems = metadata_elem.findall(".//dc:creator", namespace) or metadata_elem.findall(".//{http://purl.org/dc/elements/1.1/}creator")
                    if creator_elems:
                        authors = [elem.text for elem in creator_elems if elem.text]
                        if authors:
                            metadata["authors"] = authors
                            metadata["author"] = authors[0]
                    
                    # Extract published date
                    date_elem = metadata_elem.find(".//dc:date", namespace) or metadata_elem.find(".//{http://purl.org/dc/elements/1.1/}date")
                    if date_elem is not None and date_elem.text:
                        date_text = date_elem.text
                        metadata["date"] = date_text
                        
                        # Try to extract year
                        year_match = re.search(r"(\d{4})", date_text)
                        if year_match:
                            metadata["year"] = year_match.group(1)
                    
                    # Extract publisher
                    publisher_elem = metadata_elem.find(".//dc:publisher", namespace) or metadata_elem.find(".//{http://purl.org/dc/elements/1.1/}publisher")
                    if publisher_elem is not None and publisher_elem.text:
                        metadata["publisher"] = publisher_elem.text
                    
                    # Extract language
                    language_elem = metadata_elem.find(".//dc:language", namespace) or metadata_elem.find(".//{http://purl.org/dc/elements/1.1/}language")
                    if language_elem is not None and language_elem.text:
                        metadata["language"] = language_elem.text
                    
                    # Extract ISBN
                    identifier_elems = metadata_elem.findall(".//dc:identifier", namespace) or metadata_elem.findall(".//{http://purl.org/dc/elements/1.1/}identifier")
                    if identifier_elems:
                        for elem in identifier_elems:
                            id_text = elem.text
                            if id_text and ("isbn" in id_text.lower() or re.search(r"\d{10,13}", id_text)):
                                # Extract just the digits for ISBN
                                isbn_match = re.search(r"(?:isbn[:\s]*)?(\d[\d\-]+\d)", id_text, re.IGNORECASE)
                                if isbn_match:
                                    metadata["isbn"] = isbn_match.group(1)
                                    break
        
        except Exception as e:
            logger.error(f"Error extracting EPUB metadata: {e}")
        
        return metadata
    
    def _extract_pdf_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from PDF document info."""
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
            
        except ImportError:
            logger.warning("PyMuPDF not available for document info extraction")
        except Exception as e:
            logger.error(f"Error extracting PDF document info: {e}")
        
        return metadata
    
    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean and normalize book metadata.
        
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
            # Fix common author name formats
            # Convert "Last, First" to "First Last"
            name_parts = author.split(", ", 1)
            if len(name_parts) == 2 and not re.search(r'(Inc|Ltd|LLC|LLP|Corp|Co)', name_parts[0]):
                author = f"{name_parts[1]} {name_parts[0]}"
            cleaned["author"] = author
        
        # Clean year
        if "year" in cleaned and cleaned["year"]:
            year = cleaned["year"]
            # Extract just the 4-digit year
            year_match = re.search(r'(19\d{2}|20\d{2})', str(year))
            if year_match:
                cleaned["year"] = year_match.group(1)
        
        # Clean ISBN
        if "isbn" in cleaned and cleaned["isbn"]:
            isbn = cleaned["isbn"]
            # Remove everything except digits and hyphens
            isbn = re.sub(r'[^\d\-]', '', isbn)
            cleaned["isbn"] = isbn
        
        # Clean publisher
        if "publisher" in cleaned and cleaned["publisher"]:
            publisher = cleaned["publisher"]
            # Remove line breaks and extra spaces
            publisher = re.sub(r'\s+', ' ', publisher).strip()
            cleaned["publisher"] = publisher
        
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
        
        # Get the original extension
        original_ext = metadata.get("extension", ".epub")
        
        # Format: Author_Year_Title.ext
        if year:
            filename = f"{author}_{year}_{title}{original_ext}"
        else:
            filename = f"{author}_{title}{original_ext}"
        
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
