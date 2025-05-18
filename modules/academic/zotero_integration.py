"""
Zotero Integration Module for Academic Papers.

This module provides integration with local Zotero libraries for:
1. Comparing local PDF files with Zotero entries
2. Finding duplicates between local files and Zotero library
3. Standardizing filenames according to Zotero conventions
"""

import os
import re
import sqlite3
import logging
import hashlib
import difflib
import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from core.file_operations import FileOperations, tqdm
from modules.academic.pdf_metadata import PDFMetadataExtractor
from modules.academic.pdf_renamer import ZoteroStyleRenamer

# Setup logging
logger = logging.getLogger(__name__)

class ZoteroIntegration:
    """Integration with local Zotero library for academic papers."""
    
    def __init__(self, llm_handler=None):
        """
        Initialize Zotero integration.
        
        Args:
            llm_handler: Handler for local LLM interactions
        """
        self.llm_handler = llm_handler
        self.file_ops = FileOperations()
        self.metadata_extractor = PDFMetadataExtractor(llm_handler=llm_handler)
        self.zotero_renamer = ZoteroStyleRenamer(llm_handler=llm_handler)
        
        # Default Zotero locations
        self.zotero_db_path = None
        self.zotero_storage_path = None
        
        # Try to locate Zotero
        self._locate_zotero()
    
    def _locate_zotero(self):
        """Locate Zotero installation and database."""
        # Common Zotero locations
        possible_locations = [
            # Windows
            os.path.expanduser("~/Zotero"),
            os.path.expanduser("~/Documents/Zotero"),
            "C:/Users/tusharshah/Zotero",  # Based on path seen in previous scripts
            # macOS
            os.path.expanduser("~/Library/Application Support/Zotero"),
            # Linux
            os.path.expanduser("~/.zotero")
        ]
        
        for location in possible_locations:
            # Check for zotero.sqlite
            db_path = os.path.join(location, "zotero.sqlite")
            if os.path.isfile(db_path):
                self.zotero_db_path = db_path
                logger.info(f"Found Zotero database at {db_path}")
                
                # Look for storage folder
                storage_path = os.path.join(location, "storage")
                if os.path.isdir(storage_path):
                    self.zotero_storage_path = storage_path
                    logger.info(f"Found Zotero storage at {storage_path}")
                break
        
        if not self.zotero_db_path:
            logger.warning("Could not locate Zotero database automatically")
    
    def set_zotero_paths(self, db_path: str = None, storage_path: str = None):
        """
        Manually set Zotero paths.
        
        Args:
            db_path: Path to zotero.sqlite database
            storage_path: Path to Zotero storage folder
        """
        if db_path:
            if os.path.isfile(db_path):
                self.zotero_db_path = db_path
                logger.info(f"Zotero database path set to {db_path}")
            else:
                logger.error(f"Zotero database not found at {db_path}")
        
        if storage_path:
            if os.path.isdir(storage_path):
                self.zotero_storage_path = storage_path
                logger.info(f"Zotero storage path set to {storage_path}")
            else:
                logger.error(f"Zotero storage folder not found at {storage_path}")
    
    def _get_zotero_items(self) -> List[Dict[str, Any]]:
        """
        Get all PDF items from Zotero database.
        
        Returns:
            List of dictionaries with Zotero items
        """
        if not self.zotero_db_path:
            logger.error("Zotero database path not set")
            return []
        
        items = []
        
        try:
            # Create a copy of the database to avoid locking issues
            temp_db_path = f"{self.zotero_db_path}_temp_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
            with open(self.zotero_db_path, 'rb') as src, open(temp_db_path, 'wb') as dst:
                dst.write(src.read())
            
            # Connect to the database copy
            conn = sqlite3.connect(temp_db_path)
            cursor = conn.cursor()
            
            # Query for PDF attachments
            query = """
            SELECT i.itemID, i.key, i.dateAdded, i.dateModified, 
                   ia.path, p.libraryID, it.typeName,
                   (SELECT value FROM itemData id 
                    JOIN itemDataValues idv ON id.valueID = idv.valueID 
                    WHERE id.itemID = i.itemID AND id.fieldID = 1) as title,
                   (SELECT value FROM itemData id 
                    JOIN itemDataValues idv ON id.valueID = idv.valueID 
                    WHERE id.itemID = i.itemID AND id.fieldID = 110) as filename,
                   p.key as parentKey
            FROM items i
            JOIN itemTypes it ON i.itemTypeID = it.itemTypeID
            LEFT JOIN itemAttachments ia ON i.itemID = ia.itemID
            LEFT JOIN items p ON ia.parentItemID = p.itemID
            WHERE (ia.contentType = 'application/pdf' OR it.typeName = 'attachment') 
              AND ia.path IS NOT NULL
            """
            
            cursor.execute(query)
            
            for row in cursor.fetchall():
                item_id, key, date_added, date_modified, path, library_id, type_name, title, filename, parent_key = row
                
                # Determine file path
                attachment_path = None
                if path:
                    if path.startswith("storage:"):
                        # Storage-based path
                        rel_path = path.replace("storage:", "")
                        if self.zotero_storage_path:
                            attachment_path = os.path.join(self.zotero_storage_path, rel_path)
                    else:
                        # Absolute path or relative path
                        attachment_path = path
                
                # Only include if we have a valid path
                if attachment_path and os.path.isfile(attachment_path):
                    # Get file metadata
                    file_size = os.path.getsize(attachment_path)
                    
                    # Add to results
                    items.append({
                        "item_id": item_id,
                        "key": key,
                        "title": title or os.path.basename(attachment_path),
                        "filename": filename or os.path.basename(attachment_path),
                        "path": attachment_path,
                        "size": file_size,
                        "date_added": date_added,
                        "date_modified": date_modified,
                        "library_id": library_id,
                        "type": type_name,
                        "parent_key": parent_key
                    })
            
            conn.close()
            
            # Clean up temp db file
            os.remove(temp_db_path)
            
            logger.info(f"Found {len(items)} PDF items in Zotero database")
            return items
            
        except Exception as e:
            logger.error(f"Error accessing Zotero database: {e}")
            
            # Clean up temp db file if it exists
            if 'temp_db_path' in locals() and os.path.exists(temp_db_path):
                try:
                    os.remove(temp_db_path)
                except:
                    pass
                
            return []
    
    def scan_local_pdfs(self, directory_path: str, recursive: bool = True) -> List[Dict[str, Any]]:
        """
        Scan for local PDF files.
        
        Args:
            directory_path: Path to directory to scan
            recursive: Whether to search subdirectories
            
        Returns:
            List of dictionaries with file information
        """
        # Scan for PDF files
        pdf_paths = self.file_ops.scan_directory(
            directory_path,
            file_types=[".pdf"],
            recursive=recursive
        )
        
        logger.info(f"Found {len(pdf_paths)} PDF files in {directory_path}")
        
        # Process files with progress tracking
        local_pdfs = []
        
        for file_path in tqdm(pdf_paths, desc="Processing local PDFs"):
            try:
                file_size = os.path.getsize(file_path)
                file_name = os.path.basename(file_path)
                
                # Calculate file hash for more accurate comparison
                file_hash = self._calculate_file_hash(file_path)
                
                local_pdfs.append({
                    "path": file_path,
                    "filename": file_name,
                    "size": file_size,
                    "hash": file_hash,
                    "relative_path": os.path.relpath(file_path, directory_path)
                })
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        
        return local_pdfs
    
    def _calculate_file_hash(self, file_path: str, block_size: int = 65536) -> str:
        """
        Calculate MD5 hash of a file.
        
        Args:
            file_path: Path to the file
            block_size: Size of chunks to read
            
        Returns:
            MD5 hash as hexadecimal string
        """
        hasher = hashlib.md5()
        
        try:
            with open(file_path, 'rb') as f:
                for block in iter(lambda: f.read(block_size), b''):
                    hasher.update(block)
            return hasher.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            return ""
    
    def _normalize_filename(self, filename: str) -> str:
        """
        Normalize filename for comparison.
        
        Args:
            filename: Filename to normalize
            
        Returns:
            Normalized filename
        """
        # Remove extension
        base_name = os.path.splitext(filename)[0]
        
        # Convert to lowercase
        base_name = base_name.lower()
        
        # Remove common prefixes/suffixes
        base_name = re.sub(r'^(the|a|an)\s+', '', base_name)
        
        # Remove special characters
        base_name = re.sub(r'[^\w\s]', '', base_name)
        
        # Replace multiple spaces with single space
        base_name = re.sub(r'\s+', ' ', base_name).strip()
        
        return base_name
    
    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """
        Calculate similarity ratio between two filenames.
        
        Args:
            name1: First filename
            name2: Second filename
            
        Returns:
            Similarity ratio (0-1)
        """
        # Normalize both names
        norm1 = self._normalize_filename(name1)
        norm2 = self._normalize_filename(name2)
        
        # Calculate similarity
        return difflib.SequenceMatcher(None, norm1, norm2).ratio()
    
    def find_duplicates(self, local_pdfs: List[Dict[str, Any]], 
                      zotero_pdfs: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Find duplicates between local PDFs and Zotero library.
        
        Args:
            local_pdfs: List of local PDF information
            zotero_pdfs: List of Zotero PDF information (if None, will be fetched)
            
        Returns:
            List of potential duplicate pairs
        """
        # Get Zotero items if not provided
        if zotero_pdfs is None:
            zotero_pdfs = self._get_zotero_items()
        
        if not zotero_pdfs:
            logger.warning("No Zotero PDFs available for comparison")
            return []
        
        logger.info(f"Comparing {len(local_pdfs)} local PDFs with {len(zotero_pdfs)} Zotero PDFs")
        
        # Initialize results
        duplicates = []
        
        # Compare each local PDF with Zotero PDFs
        for local_pdf in tqdm(local_pdfs, desc="Finding duplicates"):
            local_filename = local_pdf["filename"]
            local_size = local_pdf["size"]
            local_hash = local_pdf.get("hash", "")
            
            exact_match = False
            best_match = None
            best_score = 0
            
            for zotero_pdf in zotero_pdfs:
                zotero_filename = zotero_pdf["filename"]
                zotero_size = zotero_pdf["size"]
                
                # Check for exact hash match
                if local_hash and local_hash == self._calculate_file_hash(zotero_pdf["path"]):
                    exact_match = True
                    best_match = zotero_pdf
                    best_score = 1.0
                    break
                
                # Check for exact filename match (case insensitive)
                if local_filename.lower() == zotero_filename.lower():
                    # Check if sizes are similar (within 1%)
                    if abs(local_size - zotero_size) / max(local_size, zotero_size) < 0.01:
                        exact_match = True
                        best_match = zotero_pdf
                        best_score = 1.0
                        break
                
                # Calculate filename similarity
                similarity = self._calculate_name_similarity(local_filename, zotero_filename)
                
                # If good match, keep track of best match
                if similarity > 0.6 and similarity > best_score:
                    best_match = zotero_pdf
                    best_score = similarity
            
            # If a match was found, add to duplicates
            if best_match:
                match_type = "exact_match" if exact_match else "fuzzy_name"
                
                duplicates.append({
                    "local_file": local_pdf,
                    "zotero_file": best_match,
                    "confidence": best_score,
                    "match_type": match_type
                })
        
        # Sort by confidence
        duplicates.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Log summary
        exact_matches = sum(1 for d in duplicates if d["confidence"] == 1.0)
        high_conf = sum(1 for d in duplicates if 0.8 <= d["confidence"] < 1.0)
        medium_conf = sum(1 for d in duplicates if 0.6 <= d["confidence"] < 0.8)
        
        logger.info(f"Found {len(duplicates)} potential duplicate files:")
        logger.info(f"  - {exact_matches} exact content matches")
        logger.info(f"  - {high_conf} high confidence name matches (≥80%)")
        logger.info(f"  - {medium_conf} medium confidence matches (60-79%)")
        
        return duplicates
    
    def handle_duplicate(self, duplicate: Dict[str, Any], action: str = "move_to_category") -> bool:
        """
        Handle a duplicate file.
        
        Args:
            duplicate: Duplicate information dictionary
            action: Action to take ("move_to_category", "rename_to_zotero", "skip")
            
        Returns:
            Success status
        """
        if action == "skip":
            return True
        
        local_path = duplicate["local_file"]["path"]
        confidence = duplicate["confidence"]
        
        if action == "move_to_category":
            # Determine category based on confidence
            if confidence >= 0.8:
                category = "academic_high_conf"
            else:
                category = "academic_medium_conf"
            
            # Move to appropriate to-delete folder
            return self.file_ops.safe_delete(local_path, category=category)
            
        elif action == "rename_to_zotero":
            # Rename local file to match Zotero convention
            zotero_filename = duplicate["zotero_file"]["filename"]
            new_path = os.path.join(os.path.dirname(local_path), zotero_filename)
            
            return self.file_ops.safe_move(
                local_path,
                new_path,
                category="academic",
                overwrite=False
            )
        
        return False
    
    def process_duplicates(self, duplicates: List[Dict[str, Any]], 
                         action: str = "move_to_category") -> Dict[str, int]:
        """
        Process a list of duplicates with a specified action.
        
        Args:
            duplicates: List of duplicate information dictionaries
            action: Action to take for all duplicates
            
        Returns:
            Statistics dictionary
        """
        success_count = 0
        error_count = 0
        
        for duplicate in tqdm(duplicates, desc="Processing duplicates"):
            try:
                result = self.handle_duplicate(duplicate, action=action)
                if result:
                    success_count += 1
                else:
                    error_count += 1
            except Exception as e:
                logger.error(f"Error handling duplicate: {e}")
                error_count += 1
        
        logger.info(f"Processed {len(duplicates)} duplicates: {success_count} succeeded, {error_count} failed")
        
        return {
            "total": len(duplicates),
            "success": success_count,
            "error": error_count
        }
    
    def generate_duplicate_report(self, duplicates: List[Dict[str, Any]]) -> str:
        """
        Generate a detailed report of duplicates.
        
        Args:
            duplicates: List of duplicate information dictionaries
            
        Returns:
            Path to the generated report file
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(os.getcwd(), f"duplicate_report_{timestamp}.txt")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("Duplicate PDF Files Report\n")
            f.write("=========================\n")
            f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Total potential duplicates found: {}\n\n".format(len(duplicates)))
            
            # Group by confidence
            exact_matches = [d for d in duplicates if d['confidence'] == 1.0]
            high_conf = [d for d in duplicates if d['confidence'] >= 0.8 and d['confidence'] < 1.0]
            medium_conf = [d for d in duplicates if d['confidence'] >= 0.6 and d['confidence'] < 0.8]
            
            f.write(f"Exact matches (100% confidence): {len(exact_matches)}\n")
            f.write(f"High confidence matches (≥80%): {len(high_conf)}\n")
            f.write(f"Medium confidence matches (60-79%): {len(medium_conf)}\n\n")
            
            # List all duplicates with details
            for i, dup in enumerate(duplicates, 1):
                local_file = dup['local_file']
                zotero_file = dup['zotero_file']
                
                f.write(f"{i}. Match confidence: {dup['confidence']:.2f}, Type: {dup['match_type']}\n")
                f.write(f"   Local: {local_file['filename']}\n")
                f.write(f"   Zotero: {zotero_file['filename']}\n")
                f.write(f"   Local path: {local_file['path']}\n")
                f.write(f"   Zotero path: {zotero_file['path']}\n")
                f.write(f"   Size: {local_file['size']} bytes vs {zotero_file['size']} bytes\n\n")
        
        logger.info(f"Created duplicate report: {report_file}")
        return report_file
