"""
Medical File Organization Module.

This module handles the organization of medical files based on metadata,
content type, and user-defined rules with a focus on privacy and security.
"""

import os
import getpass
import re
import logging
import datetime
from typing import Dict, Any, Tuple
from core.file_operations import FileOperations
from modules.medical.medical_metadata import MedicalMetadataExtractor

# Setup logging
logger = logging.getLogger(__name__)

class MedicalFileOrganizer:
    """Medical file organization functionality."""
    
    def __init__(self, llm_handler=None):
        """
        Initialize the medical file organizer.
        
        Args:
            llm_handler: Handler for local LLM interactions
        """
        self.llm_handler = llm_handler
        self.file_ops = FileOperations()
        self.metadata_extractor = MedicalMetadataExtractor(llm_handler=llm_handler)
        
        # Supported medical file extensions
        self.medical_extensions = [
            ".pdf", ".doc", ".docx", ".txt", ".rtf", 
            ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".gif",
            ".dicom", ".dcm", ".nii", ".nii.gz", 
            ".html", ".htm", ".xml"
        ]
        
        # Default organization structure
        self.organize_by_doc_type = True
        self.organize_by_date = True
        self.organize_by_facility = False
        self.organize_chronologically = True
        self.anonymize_filenames = True
    
    def organize_directory(self, 
                          source_dir: str, 
                          target_dir: str,
                          rename_files: bool = True,
                          recursive: bool = True,
                          dry_run: bool = False) -> Tuple[int, int, int]:
        """
        Organize medical files from source directory into target directory.
        
        Args:
            source_dir: Source directory containing medical files
            target_dir: Target directory for organized medical files
            rename_files: Whether to rename files during organization
            recursive: Whether to process subdirectories
            dry_run: If True, only show what would be done without actual changes
            
        Returns:
            Tuple of (total_files, organized_files, error_count)
        """
        # Scan for medical files
        medical_files = self.file_ops.scan_directory(
            source_dir,
            file_types=self.medical_extensions,
            recursive=recursive
        )
        
        logger.info(f"Found {len(medical_files)} medical files in {source_dir}")
        
        total_files = len(medical_files)
        organized_files = 0
        error_count = 0
        
        # Process files with progress tracking
        for file_path in medical_files:
            try:
                # Skip files that don't appear to be medical based on quick check
                if not self._is_likely_medical_file(file_path):
                    logger.info(f"Skipping non-medical file: {file_path}")
                    continue
                
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
                        category="medical",
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
        
        logger.info(f"Organized {organized_files} out of {total_files} medical files, with {error_count} errors")
        return total_files, organized_files, error_count
    
    def _is_likely_medical_file(self, file_path: str) -> bool:
        """
        Perform a quick check to determine if file is likely medical.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Whether the file is likely medical
        """
        # Check filename for common medical terms
        filename = os.path.basename(file_path).lower()
        medical_terms = [
            "lab", "test", "result", "radiology", "x-ray", "xray", "ct", "mri", 
            "ultrasound", "diagnosis", "clinical", "patient", "medical", "doctor",
            "hospital", "clinic", "health", "prescription", "medication", "treatment",
            "discharge", "summary", "pathology", "physician"
        ]
        
        # Check if any medical term is in the filename
        for term in medical_terms:
            if term in filename:
                return True
        
        # For image files, we'll need to check content
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext in [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".gif"]:
            # For image files, assume they might be medical unless proven otherwise
            # The metadata extraction will provide more certainty
            return True
        
        # For non-image files, try to check the first few lines for medical content
        # This is a basic heuristic that can be expanded
        if file_ext in [".txt", ".html", ".htm", ".xml"]:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    first_lines = "".join([next(f, "") for _ in range(10)]).lower()
                    for term in medical_terms:
                        if term in first_lines:
                            return True
            except Exception as e: 
                # If we can't read the file, assume it might be medical
                logger.warning(f"Could not read start of file {file_path} to check for medical terms: {e}")
                pass
        
        # For PDF and other binary files, we assume potentially medical
        # and will let the metadata extractor determine
        if file_ext in [".pdf", ".doc", ".docx", ".rtf", ".dicom", ".dcm", ".nii", ".nii.gz"]:
            return True
        
        # Default to not medical if no conditions matched
        return False
    
    def _get_destination_path(self, file_path: str, metadata: Dict[str, Any], target_dir: str) -> str:
        """
        Determine the destination path for a medical file based on its metadata.
        
        Args:
            file_path: Original file path
            metadata: Medical file metadata
            target_dir: Base target directory
            
        Returns:
            Full destination directory path
        """
        # Start with the target directory
        dest_path = target_dir
        
        # Extract relevant metadata
        doc_type = metadata.get("document_type", "Other")
        date_str = metadata.get("date", "")
        facility = metadata.get("facility", "")
        
        # Clean values for use in directory names
        doc_type = self._clean_for_dirname(doc_type)
        facility = self._clean_for_dirname(facility)
        
        # Build the path based on organization preferences
        
        # By document type
        if self.organize_by_doc_type and doc_type:
            dest_path = os.path.join(dest_path, doc_type)
        
        # By facility
        if self.organize_by_facility and facility:
            dest_path = os.path.join(dest_path, facility)
        
        # By date
        if self.organize_by_date and date_str:
            # Try to parse the date
            year, month = self._extract_year_month(date_str)
            if year:
                dest_path = os.path.join(dest_path, year)
                if month:
                    dest_path = os.path.join(dest_path, month)
        
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
        text = text.replace('[', '').replace(']', '').replace('(', '').replace(')', '')
        text = text.strip()
        
        # Replace spaces with underscores
        text = text.replace(' ', '_')
        
        # Remove consecutive underscores
        text = text.replace('__', '_')
        
        return text
    
    def _extract_year_month(self, date_str: str) -> Tuple[str, str]:
        """
        Extract year and month from a date string.
        
        Args:
            date_str: Date string to parse
            
        Returns:
            Tuple of (year, month) strings
        """
        # Try common formats
        date_formats = [
            ("%Y-%m-%d", r'(\d{4})-(\d{2})-\d{2}'),
            ("%m/%d/%Y", r'\d{1,2}/\d{1,2}/(\d{4})'),
            ("%d/%m/%Y", r'\d{1,2}/\d{1,2}/(\d{4})'),
            ("%B %d, %Y", r'(?:\w+)\s+\d{1,2},\s+(\d{4})'),
            ("%d %B %Y", r'\d{1,2}\s+(?:\w+)\s+(\d{4})')
        ]
        
        year = ""
        month = ""
        
        # First try to extract with regex
        for _, pattern in date_formats:
            year_match = re.search(pattern, date_str)
            if year_match:
                year = year_match.group(1)
                
                # Try to extract month if possible
                month_match = re.search(r'(?:^|[^\d])(\d{1,2})(?:[/\-])(?:\d{1,2})(?:[/\-])', date_str)
                if month_match:
                    month_num = int(month_match.group(1))
                    # Ensure month is valid
                    if 1 <= month_num <= 12:
                        # Format month as 01, 02, etc.
                        month = f"{month_num:02d}"
                
                break
        
        # If regex failed, try datetime parsing
        if not year:
            date_formats_simple = [
                "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", 
                "%Y/%m/%d", "%B %d, %Y", "%d %B %Y"
            ]
            
            for fmt in date_formats_simple:
                try:
                    date_obj = datetime.datetime.strptime(date_str, fmt)
                    year = date_obj.strftime("%Y")
                    month = date_obj.strftime("%m")
                    break
                except ValueError:
                    continue
        
        return year, month
    
    def generate_organization_report(self, 
                                   source_dir: str,
                                   target_dir: str,
                                   rename_files: bool = True,
                                   recursive: bool = True) -> str:
        """
        Generate a dry-run report of how files would be organized.
        
        Args:
            source_dir: Source directory containing medical files
            target_dir: Target directory for organized files
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
        report_file = os.path.join(os.getcwd(), f"medical_organization_report_{timestamp}.md")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Medical File Organization Plan Report\n\n")
            f.write(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("## Summary\n")
            f.write(f"- Source Directory: {source_dir}\n")
            f.write(f"- Target Directory: {target_dir}\n")
            f.write(f"- Total Files Scanned: {total}\n")
            f.write(f"- Medical Files to Organize: {organized}\n")
            f.write(f"- Potential Errors: {errors}\n\n")
            
            f.write("## Organization Settings\n")
            f.write(f"- Rename Files: {rename_files}\n")
            f.write(f"- Organize by Document Type: {self.organize_by_doc_type}\n")
            f.write(f"- Organize by Date: {self.organize_by_date}\n")
            f.write(f"- Organize by Facility: {self.organize_by_facility}\n")
            f.write(f"- Anonymize Filenames: {self.anonymize_filenames}\n\n")
            
            f.write("## File Actions Preview\n")
            f.write("The following actions would be taken when running this organization:\n\n")
            
            # Re-scan to provide detailed information
            medical_files = self.file_ops.scan_directory(
                source_dir,
                file_types=self.medical_extensions,
                recursive=recursive
            )
            
            # Only process the first 50 files for the report to avoid it getting too long
            for file_path in list(medical_files)[:50]:
                try:
                    # Check if likely medical
                    if not self._is_likely_medical_file(file_path):
                        f.write(f"- Source: {file_path}\n")
                        f.write("  - Action: Skip (Not identified as medical file)\n\n")
                        continue
                    
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
                    f.write(f"- Source: {file_path}\n")
                    f.write(f"  - Destination: {dest_file}\n")
                    f.write(f"  - Document Type: {metadata.get('document_type', 'Unknown')}\n")
                    if 'date' in metadata:
                        f.write(f"  - Date: {metadata.get('date')}\n")
                    if 'facility' in metadata:
                        f.write(f"  - Facility: {metadata.get('facility')}\n")
                    f.write("\n")
                    
                except Exception as e:
                    f.write(f"- Source: {file_path}\n")
                    f.write(f"  - Error: {str(e)}\n\n")
            
            # If there are more files than shown in the report
            if len(medical_files) > 50:
                f.write("\n*Note: Only showing the first 50 files out of {} total.*\n".format(len(medical_files)))
        
        logger.info(f"Created medical organization report: {report_file}")
        return report_file


class PrivacyAwareMedicalOrganizer(MedicalFileOrganizer):
    """Enhanced medical organizer with privacy features."""
    
    def __init__(self, llm_handler=None, config=None):
        """
        Initialize the privacy-aware medical organizer.
        
        Args:
            llm_handler: Handler for local LLM interactions
            config: Configuration dictionary
        """
        super().__init__(llm_handler=llm_handler)
        
        # Default configuration
        self.config = {
            "organize_by_doc_type": True,
            "organize_by_date": True,
            "organize_by_facility": False,
            "anonymize_filenames": True,
            "use_encryption": False,
            "privacy_level": "high",  # low, medium, high
            "track_operations": True
        }
        
        # Update with user config if provided
        if config:
            self.config.update(config)
        
        # Apply config
        self.organize_by_doc_type = self.config["organize_by_doc_type"]
        self.organize_by_date = self.config["organize_by_date"]
        self.organize_by_facility = self.config["organize_by_facility"]
        self.anonymize_filenames = self.config["anonymize_filenames"]
        self.use_encryption = self.config["use_encryption"]
        self.privacy_level = self.config["privacy_level"]
        self.track_operations = self.config["track_operations"]
        
        # Initialize operations tracker
        self.operations = []
    
    def organize_directory(self, 
                          source_dir: str, 
                          target_dir: str,
                          rename_files: bool = True,
                          recursive: bool = True,
                          dry_run: bool = False) -> Tuple[int, int, int]:
        """
        Organize medical files with privacy enhancements.
        
        Args:
            source_dir: Source directory containing medical files
            target_dir: Target directory for organized files
            rename_files: Whether to rename files during organization
            recursive: Whether to process subdirectories
            dry_run: If True, only show what would be done without actual changes
            
        Returns:
            Tuple of (total_files, organized_files, error_count)
        """
        # Reset operations tracker
        self.operations = []
        
        # Call parent method
        result = super().organize_directory(
            source_dir,
            target_dir,
            rename_files=rename_files,
            recursive=recursive,
            dry_run=dry_run
        )
        
        # Create operations log if tracking is enabled
        if self.track_operations and not dry_run:
            self._create_operations_log(target_dir)
        
        return result
    
    def _get_destination_path(self, file_path: str, metadata: Dict[str, Any], target_dir: str) -> str:
        """
        Privacy-enhanced destination path determination.
        
        Args:
            file_path: Original file path
            metadata: Medical file metadata
            target_dir: Base target directory
            
        Returns:
            Full destination directory path
        """
        # Start with standard organization
        dest_path = super()._get_destination_path(file_path, metadata, target_dir)
        
        # Add privacy enhancements based on privacy level
        if self.privacy_level == "high":
            # For high privacy, use only generic categories and date structures
            # No patient information or specific identifiers in paths
            return dest_path
        elif self.privacy_level == "medium":
            # For medium privacy, allow organization by facility but no patient info
            return dest_path
        else:  # low privacy
            # For low privacy, can add patient-specific folders if available
            if "patient_id" in metadata and metadata["patient_id"]:
                patient_id = self._clean_for_dirname(metadata["patient_id"])
                if patient_id:
                    dest_path = os.path.join(dest_path, f"Patient_{patient_id}")
            
            return dest_path
    
    def _encrypt_file(self, source_file: str, dest_file: str) -> bool:
        """
        Encrypt a file using simple password-based encryption.
        Requires the cryptography package.
        
        Args:
            source_file: Source file path
            dest_file: Destination file path
            
        Returns:
            Success status
        """
        if not self.use_encryption:
            # Just copy the file if encryption is disabled
            return self.file_ops.safe_move(source_file, dest_file, category="medical", overwrite=False)
        
        try:
            # Try to import cryptography
            from cryptography.fernet import Fernet
            import base64
            import hashlib
            
            password = getpass.getpass(prompt="Enter encryption password: ")
            if not password:
                logger.error("No password provided. Encryption aborted.")
                return False
            
            # Generate a key from the password
            key = base64.urlsafe_b64encode(hashlib.sha256(password.encode()).digest())
            fernet = Fernet(key)
            
            # Read the file
            with open(source_file, "rb") as f:
                file_data = f.read()
            
            # Encrypt the data
            encrypted_data = fernet.encrypt(file_data)
            
            # Write to the destination
            os.makedirs(os.path.dirname(dest_file), exist_ok=True)
            with open(dest_file, "wb") as f:
                f.write(encrypted_data)
            
            # Track the operation
            if self.track_operations:
                self.operations.append({
                    "source": source_file,
                    "destination": dest_file,
                    "action": "encrypt",
                    "timestamp": datetime.datetime.now().isoformat()
                })
            
            return True
            
        except ImportError:
            logger.warning("Cryptography package not available. Falling back to regular file move.")
            return self.file_ops.safe_move(source_file, dest_file, category="medical", overwrite=False)
        except Exception as e:
            logger.error(f"Error encrypting file: {e}")
            return False
    
    def _create_operations_log(self, target_dir: str) -> str:
        """
        Create a log of all operations performed.
        
        Args:
            target_dir: Base target directory
            
        Returns:
            Path to the log file
        """
        # Create a unique timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(target_dir, f"medical_operations_log_{timestamp}.csv")
        
        try:
            # Create the target directory if it doesn't exist
            os.makedirs(target_dir, exist_ok=True)
            
            # Write operations as CSV
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write("timestamp,source,destination,action\n")
                for op in self.operations:
                    f.write(f"{op['timestamp']},{op['source']},{op['destination']},{op['action']}\n")
            
            logger.info(f"Created operations log: {log_file}")
            return log_file
            
        except Exception as e:
            logger.error(f"Error creating operations log: {e}")
            return ""
