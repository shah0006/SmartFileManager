"""
General File Organization Module.

This module handles the organization of general files into appropriate
directories based on file type, content, and user-defined rules.
"""

import os
import re
import logging
import datetime
import shutil
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path

from core.file_operations import FileOperations
from modules.general.file_analyzer import FileAnalyzer

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    from core.file_operations import tqdm

# Setup logging
logger = logging.getLogger(__name__)

class FileOrganizer:
    """General file organization functionality."""
    
    def __init__(self, llm_handler=None):
        """
        Initialize the file organizer.
        
        Args:
            llm_handler: Handler for local LLM interactions
        """
        self.llm_handler = llm_handler
        self.file_ops = FileOperations()
        self.file_analyzer = FileAnalyzer(llm_handler=llm_handler)
        
        # Default organization structure
        self.organize_by_type = True
        self.organize_by_date = False
        self.create_type_folders = True
        self.exclude_extensions = set(['.ini', '.lnk', '.url', '.tmp', '.temp', '.log', '.sys'])
        
        # List of extensions to process
        self.supported_extensions = set()
        for ext_list in self.file_analyzer.category_mapping.values():
            self.supported_extensions.update(ext_list)
    
    def organize_directory(self, 
                          source_dir: str, 
                          target_dir: str,
                          recursive: bool = True,
                          dry_run: bool = False,
                          exclude_dirs: Optional[List[str]] = None) -> Tuple[int, int, int]:
        """
        Organize files from source directory into target directory.
        
        Args:
            source_dir: Source directory containing files
            target_dir: Target directory for organized files
            recursive: Whether to process subdirectories
            dry_run: If True, only show what would be done without actual changes
            exclude_dirs: List of directory names to exclude (e.g., 'System Volume Information')
            
        Returns:
            Tuple of (total_files, organized_files, error_count)
        """
        # Default excludes if none provided
        if exclude_dirs is None:
            exclude_dirs = ['System Volume Information', '$RECYCLE.BIN', '.Trash', 
                            '.git', 'node_modules', '__pycache__', '.vscode']
        
        # Scan directory for files
        all_files = []
        if recursive:
            for root, dirs, files in os.walk(source_dir):
                # Skip excluded directories
                dirs[:] = [d for d in dirs if d not in exclude_dirs]
                
                for file in files:
                    file_path = os.path.join(root, file)
                    all_files.append(file_path)
        else:
            for file in os.listdir(source_dir):
                file_path = os.path.join(source_dir, file)
                if os.path.isfile(file_path):
                    all_files.append(file_path)
        
        # Filter files by extension
        included_files = []
        for file_path in all_files:
            ext = os.path.splitext(file_path)[1].lower()
            if ext not in self.exclude_extensions:
                included_files.append(file_path)
        
        logger.info(f"Found {len(included_files)} files to organize in {source_dir}")
        
        total_files = len(included_files)
        organized_files = 0
        error_count = 0
        
        # Process files with progress tracking
        for file_path in tqdm(included_files, desc="Organizing files"):
            try:
                # Categorize the file
                category, metadata = self.file_analyzer.categorize_file(file_path)
                
                # Determine destination path
                dest_path = self._get_destination_path(file_path, category, metadata, target_dir)
                
                # Determine destination filename
                dest_file = os.path.join(dest_path, os.path.basename(file_path))
                
                # Handle potential filename conflicts
                dest_file = self._get_unique_filename(dest_file)
                
                # Log the planned action
                logger.info(f"{'Would move' if dry_run else 'Moving'}: {file_path} -> {dest_file}")
                
                if not dry_run:
                    # Ensure destination directory exists
                    os.makedirs(dest_path, exist_ok=True)
                    
                    # Move the file
                    result = self.file_ops.safe_move(
                        file_path,
                        dest_file,
                        category="general",
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
        
        logger.info(f"Organized {organized_files} out of {total_files} files, with {error_count} errors")
        return total_files, organized_files, error_count
    
    def _get_destination_path(self, file_path: str, category: str, metadata: Dict[str, Any], target_dir: str) -> str:
        """
        Determine the destination path for a file.
        
        Args:
            file_path: Original file path
            category: File category
            metadata: File metadata
            target_dir: Base target directory
            
        Returns:
            Full destination directory path
        """
        # Start with the target directory
        dest_path = target_dir
        
        # Organize by type if enabled
        if self.organize_by_type and category:
            if self.create_type_folders:
                # Organize into type-specific subdirectories
                dest_path = os.path.join(dest_path, category.capitalize())
                
                # For certain categories, add subcategories
                if category == 'document':
                    ext = os.path.splitext(file_path)[1].lower()
                    if ext == '.pdf':
                        dest_path = os.path.join(dest_path, 'PDF')
                    elif ext in ['.doc', '.docx']:
                        dest_path = os.path.join(dest_path, 'Word')
                    elif ext in ['.txt', '.md']:
                        dest_path = os.path.join(dest_path, 'Text')
                elif category == 'image':
                    # Try to determine if it's a photo or graphic
                    if self._is_likely_photo(file_path, metadata):
                        dest_path = os.path.join(dest_path, 'Photos')
                    else:
                        dest_path = os.path.join(dest_path, 'Graphics')
        
        # Organize by date if enabled
        if self.organize_by_date and 'modified' in metadata:
            try:
                # Parse the date
                date_obj = datetime.datetime.fromisoformat(metadata['modified'])
                year = str(date_obj.year)
                month = f"{date_obj.month:02d} - {date_obj.strftime('%b')}"
                
                # Add to path
                dest_path = os.path.join(dest_path, year, month)
            except Exception as e:
                logger.error(f"Error parsing date for {file_path}: {e}")
        
        return dest_path
    
    def _get_unique_filename(self, file_path: str) -> str:
        """
        Generate a unique filename if the destination already exists.
        
        Args:
            file_path: Destination file path
            
        Returns:
            Unique file path
        """
        if not os.path.exists(file_path):
            return file_path
        
        path, name = os.path.split(file_path)
        name, ext = os.path.splitext(name)
        
        # Try adding a counter
        counter = 1
        while True:
            new_path = os.path.join(path, f"{name} ({counter}){ext}")
            if not os.path.exists(new_path):
                return new_path
            counter += 1
    
    def _is_likely_photo(self, file_path: str, metadata: Dict[str, Any]) -> bool:
        """
        Determine if an image is likely a photo vs. a graphic.
        
        Args:
            file_path: Path to the image file
            metadata: File metadata
            
        Returns:
            True if likely a photo, False otherwise
        """
        # Check file extension
        ext = os.path.splitext(file_path)[1].lower()
        photo_extensions = ['.jpg', '.jpeg', '.heic', '.cr2', '.nef', '.arw', '.dng']
        
        if ext in photo_extensions:
            return True
        
        # Check file size (photos tend to be larger)
        if 'size' in metadata and metadata['size'] > 100000:  # > 100KB
            return True
        
        # More complex checks could include image analysis with PIL
        # But that would be too expensive for a quick organizer
        
        return False
    
    def generate_organization_report(self, 
                                   source_dir: str,
                                   target_dir: str,
                                   recursive: bool = True) -> str:
        """
        Generate a dry-run report of how files would be organized.
        
        Args:
            source_dir: Source directory containing files
            target_dir: Target directory for organized files
            recursive: Whether to process subdirectories
            
        Returns:
            Path to the generated report file
        """
        # Perform a dry run organization
        total, organized, errors = self.organize_directory(
            source_dir,
            target_dir,
            recursive=recursive,
            dry_run=True
        )
        
        # Create a report
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(os.getcwd(), f"file_organization_report_{timestamp}.md")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# File Organization Plan Report\n\n")
            f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"## Summary\n")
            f.write(f"- Source Directory: `{source_dir}`\n")
            f.write(f"- Target Directory: `{target_dir}`\n")
            f.write(f"- Total Files: {total}\n")
            f.write(f"- Files to Organize: {organized}\n")
            f.write(f"- Potential Errors: {errors}\n\n")
            
            f.write(f"## Organization Settings\n")
            f.write(f"- Organize by Type: {self.organize_by_type}\n")
            f.write(f"- Organize by Date: {self.organize_by_date}\n")
            f.write(f"- Create Type Folders: {self.create_type_folders}\n\n")
            
            f.write(f"## Category Distribution\n")
            f.write("Category counts will be calculated during the actual organization process.\n\n")
            
            f.write(f"## Recommended Next Steps\n")
            f.write("1. Review this report to ensure the organization plan meets your needs.\n")
            f.write("2. Run the organizer with dry_run=False to perform the actual organization.\n")
            f.write("3. After organization, check for any files that weren't organized as expected.\n")
        
        logger.info(f"Created organization report: {report_file}")
        return report_file


class SmartFileOrganizer(FileOrganizer):
    """Enhanced file organizer with content analysis capabilities."""
    
    def __init__(self, llm_handler=None, config=None):
        """
        Initialize the smart file organizer.
        
        Args:
            llm_handler: Handler for local LLM interactions
            config: Configuration dictionary
        """
        super().__init__(llm_handler=llm_handler)
        
        # Default configuration
        self.config = {
            "organize_by_type": True,
            "organize_by_date": False,
            "create_type_folders": True,
            "organize_by_content": True,
            "handle_downloads": True,
            "clean_empty_dirs": True,
            "group_related_files": True
        }
        
        # Update with user config if provided
        if config:
            self.config.update(config)
        
        # Apply config
        self.organize_by_type = self.config["organize_by_type"]
        self.organize_by_date = self.config["organize_by_date"]
        self.create_type_folders = self.config["create_type_folders"]
        self.organize_by_content = self.config["organize_by_content"]
        self.handle_downloads = self.config["handle_downloads"]
        self.clean_empty_dirs = self.config["clean_empty_dirs"]
        self.group_related_files = self.config["group_related_files"]
        
        # Additional organization rules
        self.downloads_rules = [
            (r'\.pkg$|\.dmg$|\.exe$|\.msi$|\.app$', 'Installers'),
            (r'\.zip$|\.rar$|\.7z$|\.tar\.gz$', 'Archives'),
            (r'\.pdf$|\.docx?$|\.xlsx?$|\.pptx?$', 'Documents'),
            (r'\.jpg$|\.jpeg$|\.png$|\.gif$', 'Images'),
            (r'\.mp4$|\.mov$|\.avi$|\.mkv$', 'Videos'),
            (r'\.mp3$|\.wav$|\.flac$|\.aac$', 'Audio')
        ]
    
    def _get_destination_path(self, file_path: str, category: str, metadata: Dict[str, Any], target_dir: str) -> str:
        """
        Enhanced destination path determination using content analysis.
        
        Args:
            file_path: Original file path
            category: File category
            metadata: File metadata
            target_dir: Base target directory
            
        Returns:
            Full destination directory path
        """
        # Start with standard organization
        dest_path = super()._get_destination_path(file_path, category, metadata, target_dir)
        
        # If this is a downloads folder and we should handle downloads specially
        if self.handle_downloads and "download" in source_dir.lower():
            dest_path = self._apply_downloads_rules(file_path, target_dir)
        
        # Content-based organization if enabled and LLM is available
        if self.organize_by_content and self.llm_handler:
            if category == 'document' and metadata.get('first_page_text') or metadata.get('sample_text'):
                # Get content text
                content = metadata.get('first_page_text') or metadata.get('sample_text') or ""
                
                if content:
                    topic = self._extract_document_topic(content)
                    if topic:
                        # Add topic folder
                        dest_path = os.path.join(dest_path, self._clean_for_dirname(topic))
        
        return dest_path
    
    def _apply_downloads_rules(self, file_path: str, target_dir: str) -> str:
        """
        Apply special rules for Downloads folder.
        
        Args:
            file_path: Original file path
            target_dir: Base target directory
            
        Returns:
            Destination path
        """
        filename = os.path.basename(file_path).lower()
        
        for pattern, folder in self.downloads_rules:
            if re.search(pattern, filename):
                return os.path.join(target_dir, "Downloads", folder)
        
        # If no rule matched, use a catch-all folder
        return os.path.join(target_dir, "Downloads", "Other")
    
    def _extract_document_topic(self, content: str) -> Optional[str]:
        """
        Extract document topic using LLM.
        
        Args:
            content: Document content sample
            
        Returns:
            Topic classification or None
        """
        if not self.llm_handler:
            return None
        
        try:
            # Prepare a prompt for topic extraction
            prompt = f"""
            Based on the following document excerpt, identify the main topic or subject category.
            Provide a single, general category like "Finance", "Technology", "Education", "Health", etc.
            
            Document excerpt:
            {content[:1000]}
            
            Respond with just the topic name.
            """
            
            # Get response
            topic = self.llm_handler.get_response(prompt).strip()
            
            logger.info(f"Extracted topic: {topic}")
            return topic
            
        except Exception as e:
            logger.error(f"Error extracting document topic: {e}")
            return None
    
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
    
    def clean_empty_directories(self, directory: str) -> int:
        """
        Remove empty directories after organization.
        
        Args:
            directory: Directory to clean
            
        Returns:
            Number of directories removed
        """
        if not self.clean_empty_dirs:
            return 0
        
        count = 0
        
        try:
            for root, dirs, files in os.walk(directory, topdown=False):
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    
                    # Check if directory is empty
                    if not os.listdir(dir_path):
                        logger.info(f"Removing empty directory: {dir_path}")
                        os.rmdir(dir_path)
                        count += 1
            
            return count
            
        except Exception as e:
            logger.error(f"Error cleaning empty directories: {e}")
            return count
    
    def group_related_files(self, directory: str) -> int:
        """
        Group related files together (e.g., same basename but different extensions).
        
        Args:
            directory: Directory to process
            
        Returns:
            Number of files grouped
        """
        if not self.group_related_files:
            return 0
        
        count = 0
        
        try:
            # Find all files
            all_files = []
            for root, _, files in os.walk(directory):
                for file in files:
                    all_files.append(os.path.join(root, file))
            
            # Group by basename (without extension)
            basename_map = {}
            for file_path in all_files:
                basename = os.path.splitext(os.path.basename(file_path))[0]
                if basename not in basename_map:
                    basename_map[basename] = []
                basename_map[basename].append(file_path)
            
            # Process groups with multiple files
            for basename, file_group in basename_map.items():
                if len(file_group) > 1:
                    # Create a subdirectory for the group
                    group_dir = os.path.join(os.path.dirname(file_group[0]), basename)
                    os.makedirs(group_dir, exist_ok=True)
                    
                    # Move all files to the group directory
                    for file_path in file_group:
                        new_path = os.path.join(group_dir, os.path.basename(file_path))
                        if file_path != new_path:  # Skip if it's already in the right place
                            if not os.path.exists(new_path):
                                shutil.move(file_path, new_path)
                                count += 1
            
            return count
            
        except Exception as e:
            logger.error(f"Error grouping related files: {e}")
            return count
