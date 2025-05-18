"""
Book Organization Module.

This module handles the organization of book files into appropriate
directories based on metadata, file type, and user-defined rules.
"""

import os
import re
import logging
import datetime
import shutil
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from core.file_operations import FileOperations
from modules.books.book_metadata import BookMetadataExtractor
from modules.books.book_renamer import BookRenamer

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    from core.file_operations import tqdm

# Setup logging
logger = logging.getLogger(__name__)

class BookOrganizer:
    """Book organization functionality."""
    
    def __init__(self, llm_handler=None):
        """
        Initialize the book organizer.
        
        Args:
            llm_handler: Handler for local LLM interactions
        """
        self.llm_handler = llm_handler
        self.file_ops = FileOperations()
        self.metadata_extractor = BookMetadataExtractor(llm_handler=llm_handler)
        self.book_renamer = BookRenamer(llm_handler=llm_handler)
        
        # Supported book extensions
        self.book_extensions = [".epub", ".mobi", ".azw", ".azw3", ".pdf", ".txt"]
        
        # Default organization structure
        self.organize_by_author = True
        self.organize_by_year = False
        self.organize_by_publisher = False
        self.organize_by_language = False
        self.create_alphabetical_dirs = True
    
    def organize_directory(self, 
                          source_dir: str, 
                          target_dir: str,
                          rename_files: bool = True,
                          recursive: bool = True,
                          dry_run: bool = False) -> Tuple[int, int, int]:
        """
        Organize book files from source directory into target directory.
        
        Args:
            source_dir: Source directory containing book files
            target_dir: Target directory for organized books
            rename_files: Whether to rename files during organization
            recursive: Whether to process subdirectories
            dry_run: If True, only show what would be done without actual changes
            
        Returns:
            Tuple of (total_files, organized_files, error_count)
        """
        # Scan for book files
        book_files = self.file_ops.scan_directory(
            source_dir,
            file_types=self.book_extensions,
            recursive=recursive
        )
        
        logger.info(f"Found {len(book_files)} book files in {source_dir}")
        
        total_files = len(book_files)
        organized_files = 0
        error_count = 0
        
        # Process files with progress tracking
        for file_path in tqdm(book_files, desc="Organizing books"):
            try:
                # Extract metadata
                metadata = self.metadata_extractor.extract_metadata(file_path)
                
                # Determine destination path
                dest_path = self._get_destination_path(file_path, metadata, target_dir)
                
                # Generate new filename if rename is enabled
                if rename_files:
                    new_filename = self.metadata_extractor.generate_filename(metadata)
                    dest_file = os.path.join(dest_path, new_filename)
                else:
                    dest_file = os.path.join(dest_path, os.path.basename(file_path))
                
                # Log the planned action
                logger.info(f"{'Would move' if dry_run else 'Moving'}: {file_path} -> {dest_file}")
                
                if not dry_run:
                    # Ensure destination directory exists
                    os.makedirs(dest_path, exist_ok=True)
                    
                    # Move the file
                    result = self.file_ops.safe_move(
                        file_path,
                        dest_file,
                        category="books",
                        overwrite=False
                    )
                    
                    if result:
                        organized_files += 1
                    else:
                        error_count += 1
                else:
                    # In dry run mode, consider this a success
                    organized_files += 1
                
            except Exception as e:
                logger.error(f"Error organizing {file_path}: {e}")
                error_count += 1
        
        logger.info(f"Organized {organized_files} out of {total_files} books, with {error_count} errors")
        return total_files, organized_files, error_count
    
    def _get_destination_path(self, file_path: str, metadata: Dict[str, Any], target_dir: str) -> str:
        """
        Determine the destination path for a book based on its metadata.
        
        Args:
            file_path: Original file path
            metadata: Book metadata
            target_dir: Base target directory
            
        Returns:
            Full destination directory path
        """
        # Start with the target directory
        dest_path = target_dir
        
        # Extract relevant metadata
        extension = os.path.splitext(file_path)[1].lower()
        author = metadata.get("author", "Unknown Author")
        year = metadata.get("year", "")
        publisher = metadata.get("publisher", "")
        language = metadata.get("language", "")
        
        # Clean values for use in directory names
        author = self._clean_for_dirname(author)
        publisher = self._clean_for_dirname(publisher)
        language = self._clean_for_dirname(language)
        
        # Build the path based on organization preferences
        
        # By file type
        file_type_dir = extension[1:].upper()  # Remove dot, convert to uppercase
        dest_path = os.path.join(dest_path, file_type_dir)
        
        # By alphabetical directory
        if self.create_alphabetical_dirs and author and author[0].isalpha():
            alpha_dir = author[0].upper()
            dest_path = os.path.join(dest_path, alpha_dir)
        
        # By author
        if self.organize_by_author and author:
            dest_path = os.path.join(dest_path, author)
        
        # By year
        if self.organize_by_year and year:
            dest_path = os.path.join(dest_path, year)
        
        # By publisher
        if self.organize_by_publisher and publisher:
            dest_path = os.path.join(dest_path, publisher)
        
        # By language
        if self.organize_by_language and language:
            dest_path = os.path.join(dest_path, language)
        
        return dest_path
    
    def _clean_for_dirname(self, text: str) -> str:
        """
        Clean text for use in a directory name.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text suitable for directory names
        """
        if not text:
            return ""
        
        # Replace spaces and special characters
        text = re.sub(r'[^\w\s\-]', '', text)
        text = text.strip()
        
        # Replace spaces with underscores
        text = re.sub(r'\s+', '_', text)
        
        # Remove consecutive underscores
        text = re.sub(r'_+', '_', text)
        
        return text
    
    def generate_organization_report(self, 
                                   source_dir: str,
                                   target_dir: str,
                                   rename_files: bool = True,
                                   recursive: bool = True) -> str:
        """
        Generate a dry-run report of how files would be organized.
        
        Args:
            source_dir: Source directory containing book files
            target_dir: Target directory for organized books
            rename_files: Whether files would be renamed
            recursive: Whether to process subdirectories
            
        Returns:
            Path to the generated report file
        """
        # Perform a dry run organization
        total, organized, errors = self.organize_directory(
            source_dir,
            target_dir,
            rename_files=rename_files,
            recursive=recursive,
            dry_run=True
        )
        
        # Create a report
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(os.getcwd(), f"book_organization_report_{timestamp}.md")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Book Organization Plan Report\n\n")
            f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"## Summary\n")
            f.write(f"- Source Directory: `{source_dir}`\n")
            f.write(f"- Target Directory: `{target_dir}`\n")
            f.write(f"- Total Book Files: {total}\n")
            f.write(f"- Files to Organize: {organized}\n")
            f.write(f"- Potential Errors: {errors}\n\n")
            
            f.write(f"## Organization Settings\n")
            f.write(f"- Rename Files: {rename_files}\n")
            f.write(f"- Organize by Author: {self.organize_by_author}\n")
            f.write(f"- Organize by Year: {self.organize_by_year}\n")
            f.write(f"- Organize by Publisher: {self.organize_by_publisher}\n")
            f.write(f"- Organize by Language: {self.organize_by_language}\n")
            f.write(f"- Create Alphabetical Directories: {self.create_alphabetical_dirs}\n\n")
            
            f.write(f"## File Actions Preview\n")
            f.write(f"The following actions would be taken when running this organization:\n\n")
            
            # Re-scan to provide detailed information
            book_files = self.file_ops.scan_directory(
                source_dir,
                file_types=self.book_extensions,
                recursive=recursive
            )
            
            for file_path in book_files:
                try:
                    # Extract metadata
                    metadata = self.metadata_extractor.extract_metadata(file_path)
                    
                    # Determine destination path
                    dest_path = self._get_destination_path(file_path, metadata, target_dir)
                    
                    # Generate new filename if rename is enabled
                    if rename_files:
                        new_filename = self.metadata_extractor.generate_filename(metadata)
                        dest_file = os.path.join(dest_path, new_filename)
                    else:
                        dest_file = os.path.join(dest_path, os.path.basename(file_path))
                    
                    # Add to report
                    f.write(f"- **Source:** `{file_path}`\n")
                    f.write(f"  - **Destination:** `{dest_file}`\n")
                    f.write(f"  - **Title:** {metadata.get('title', 'Unknown')}\n")
                    f.write(f"  - **Author:** {metadata.get('author', 'Unknown')}\n")
                    if 'year' in metadata:
                        f.write(f"  - **Year:** {metadata.get('year')}\n")
                    f.write(f"\n")
                    
                except Exception as e:
                    f.write(f"- **Source:** `{file_path}`\n")
                    f.write(f"  - **Error:** {str(e)}\n\n")
        
        logger.info(f"Created organization report: {report_file}")
        return report_file


class SmartBookOrganizer(BookOrganizer):
    """Enhanced book organizer with content-based organization capabilities."""
    
    def __init__(self, llm_handler=None, config=None):
        """
        Initialize the smart book organizer.
        
        Args:
            llm_handler: Handler for local LLM interactions
            config: Configuration dictionary
        """
        super().__init__(llm_handler=llm_handler)
        
        # Default configuration
        self.config = {
            "organize_by_author": True,
            "organize_by_year": False,
            "organize_by_publisher": False,
            "organize_by_language": False,
            "create_alphabetical_dirs": True,
            "organize_by_genre": False,
            "organize_by_topics": False,
            "min_confidence": 0.7
        }
        
        # Update with user config if provided
        if config:
            self.config.update(config)
        
        # Apply config
        self.organize_by_author = self.config["organize_by_author"]
        self.organize_by_year = self.config["organize_by_year"]
        self.organize_by_publisher = self.config["organize_by_publisher"]
        self.organize_by_language = self.config["organize_by_language"]
        self.create_alphabetical_dirs = self.config["create_alphabetical_dirs"]
        self.organize_by_genre = self.config["organize_by_genre"]
        self.organize_by_topics = self.config["organize_by_topics"]
    
    def _get_destination_path(self, file_path: str, metadata: Dict[str, Any], target_dir: str) -> str:
        """
        Enhanced destination path determination using content-based classification.
        
        Args:
            file_path: Original file path
            metadata: Book metadata
            target_dir: Base target directory
            
        Returns:
            Full destination directory path
        """
        # Start with standard organization
        dest_path = super()._get_destination_path(file_path, metadata, target_dir)
        
        # Add content-based organization if enabled and LLM is available
        if self.llm_handler:
            # By genre
            if self.organize_by_genre and metadata.get("content_sample"):
                genre = self._classify_genre(metadata.get("content_sample", ""), metadata)
                if genre:
                    dest_path = os.path.join(dest_path, self._clean_for_dirname(genre))
            
            # By topics/subjects
            if self.organize_by_topics and metadata.get("content_sample"):
                topics = self._extract_topics(metadata.get("content_sample", ""), metadata)
                if topics and len(topics) > 0:
                    primary_topic = topics[0]
                    dest_path = os.path.join(dest_path, self._clean_for_dirname(primary_topic))
        
        return dest_path
    
    def _classify_genre(self, content: str, metadata: Dict[str, Any]) -> Optional[str]:
        """
        Classify book genre using LLM.
        
        Args:
            content: Book content sample
            metadata: Book metadata
            
        Returns:
            Genre classification or None
        """
        if not self.llm_handler:
            return None
        
        try:
            # Prepare a prompt for genre classification
            title = metadata.get("title", "Unknown")
            author = metadata.get("author", "Unknown")
            
            prompt = f"""
            Classify the genre of this book based on the metadata and content sample.
            Title: {title}
            Author: {author}
            
            Content sample:
            {content[:2000]}
            
            Provide exactly one genre from this list: Fiction, Non-Fiction, Science Fiction, 
            Fantasy, Mystery, Thriller, Romance, Historical, Biography, Science, 
            Technology, Philosophy, Psychology, Self-Help, Business, Academic
            
            Respond with just the genre name.
            """
            
            # Get response
            genre = self.llm_handler.get_response(prompt).strip()
            
            logger.info(f"Classified genre for '{title}': {genre}")
            return genre
            
        except Exception as e:
            logger.error(f"Error classifying genre: {e}")
            return None
    
    def _extract_topics(self, content: str, metadata: Dict[str, Any]) -> List[str]:
        """
        Extract main topics using LLM.
        
        Args:
            content: Book content sample
            metadata: Book metadata
            
        Returns:
            List of main topics
        """
        if not self.llm_handler:
            return []
        
        try:
            # Prepare a prompt for topic extraction
            title = metadata.get("title", "Unknown")
            author = metadata.get("author", "Unknown")
            
            prompt = f"""
            Extract the main topics or subjects of this book based on the metadata and content sample.
            Title: {title}
            Author: {author}
            
            Content sample:
            {content[:2000]}
            
            Provide up to three main topics/subjects, separated by commas.
            Respond with just the topic list.
            """
            
            # Get response
            response = self.llm_handler.get_response(prompt).strip()
            
            # Parse topics
            topics = [topic.strip() for topic in response.split(",")]
            topics = [topic for topic in topics if topic]
            
            logger.info(f"Extracted topics for '{title}': {topics}")
            return topics
            
        except Exception as e:
            logger.error(f"Error extracting topics: {e}")
            return []
