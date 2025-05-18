#!/usr/bin/env python3
"""
Book Module Test Script for SmartFileManager

This script performs unit tests for the book module, including
metadata extraction, renaming, and organization functionality.
"""

import sys
import os
import time
import logging
import argparse
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add parent directory to path so we can import from the project
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import book module
try:
    from modules.books.book_metadata import BookMetadataExtractor
    from modules.books.book_renamer import BookRenamer, SmartBookRenamer
    from modules.books.book_organizer import BookOrganizer
    HAS_BOOK_MODULES = True
except ImportError:
    HAS_BOOK_MODULES = False

# Import utilities
from utils.llm_handler import get_llm_handler
from config.config import get_config_manager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("book-module-test")

class BookModuleTest:
    """Test class for book module functionality."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the book module test.
        
        Args:
            config_file: Optional path to custom config file
        """
        if not HAS_BOOK_MODULES:
            raise ImportError("Book modules not found. Make sure they are properly installed.")
        
        # Initialize configuration
        self.config_manager = get_config_manager(config_file)
        self.llm_config = self.config_manager.get_config("llm")
        self.book_config = self.config_manager.get_config("books")
        
        # Initialize LLM handler
        self.llm_handler = get_llm_handler(self.llm_config)
        
        # Initialize book modules
        self.metadata_extractor = BookMetadataExtractor(llm_handler=self.llm_handler)
        self.book_renamer = BookRenamer(llm_handler=self.llm_handler)
        self.smart_renamer = SmartBookRenamer(llm_handler=self.llm_handler)
        self.book_organizer = BookOrganizer(llm_handler=self.llm_handler)
        
        # Configure organizer from settings
        self.book_organizer.organize_by_author = self.book_config.get("organize_by_author", True)
        self.book_organizer.organize_by_year = self.book_config.get("organize_by_year", False)
        
        # Set up test results
        self.results = {
            "metadata_extraction": [],
            "filename_generation": [],
            "renaming": [],
            "organization": []
        }
    
    def test_metadata_extraction(self, file_path: str) -> Dict[str, Any]:
        """
        Test metadata extraction on a book file.
        
        Args:
            file_path: Path to the book file
            
        Returns:
            Test result dictionary
        """
        logger.info(f"Testing metadata extraction on: {file_path}")
        
        result = {
            "file": os.path.basename(file_path),
            "success": False,
            "metadata": None,
            "time_taken": None,
            "error": None
        }
        
        try:
            start_time = time.time()
            metadata = self.metadata_extractor.extract_metadata(file_path)
            elapsed_time = time.time() - start_time
            
            result["success"] = bool(metadata)
            result["metadata"] = metadata
            result["time_taken"] = elapsed_time
            
            logger.info(f"Metadata extraction {'succeeded' if result['success'] else 'failed'} in {elapsed_time:.2f}s")
            
            # Add to results
            self.results["metadata_extraction"].append(result)
            
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            result["error"] = str(e)
            self.results["metadata_extraction"].append(result)
        
        return result
    
    def test_filename_generation(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test filename generation from metadata.
        
        Args:
            metadata: Metadata dictionary
            
        Returns:
            Test result dictionary
        """
        logger.info("Testing filename generation")
        
        result = {
            "success": False,
            "standard_filename": None,
            "smart_filename": None,
            "error": None
        }
        
        try:
            # Standard filename generation
            standard_filename = self.metadata_extractor.generate_filename(metadata)
            result["standard_filename"] = standard_filename
            
            # Smart filename generation
            smart_filename = self.smart_renamer.generate_filename(metadata)
            result["smart_filename"] = smart_filename
            
            result["success"] = bool(standard_filename and smart_filename)
            
            logger.info(f"Filename generation {'succeeded' if result['success'] else 'failed'}")
            logger.info(f"Standard filename: {standard_filename}")
            logger.info(f"Smart filename: {smart_filename}")
            
            # Add to results
            self.results["filename_generation"].append(result)
            
        except Exception as e:
            logger.error(f"Error generating filename: {e}")
            result["error"] = str(e)
            self.results["filename_generation"].append(result)
        
        return result
    
    def test_renaming(self, source_dir: str, test_files: List[str]) -> Dict[str, Any]:
        """
        Test book renaming functionality.
        
        Args:
            source_dir: Directory containing test files
            test_files: List of test files to rename
            
        Returns:
            Test result dictionary
        """
        logger.info(f"Testing book renaming on {len(test_files)} files")
        
        result = {
            "total_files": len(test_files),
            "renamed_files": 0,
            "errors": 0,
            "time_taken": None,
            "details": []
        }
        
        try:
            # Create a temporary directory for testing
            with tempfile.TemporaryDirectory() as temp_dir:
                # Copy test files to temporary directory
                for file_path in test_files:
                    shutil.copy2(file_path, temp_dir)
                
                # Run book renamer on temporary directory
                start_time = time.time()
                total, renamed, errors = self.book_renamer.process_directory(
                    temp_dir,
                    recursive=False,
                    rename_immediately=True
                )
                elapsed_time = time.time() - start_time
                
                result["renamed_files"] = renamed
                result["errors"] = errors
                result["time_taken"] = elapsed_time
                
                # Get details for renamed files
                for file in os.listdir(temp_dir):
                    file_path = os.path.join(temp_dir, file)
                    result["details"].append({
                        "file": file,
                        "renamed": file != os.path.basename(test_files[0])
                    })
                
                logger.info(f"Renamed {renamed} out of {total} files with {errors} errors in {elapsed_time:.2f}s")
            
            # Add to results
            self.results["renaming"].append(result)
            
        except Exception as e:
            logger.error(f"Error in renaming test: {e}")
            result["error"] = str(e)
            self.results["renaming"].append(result)
        
        return result
    
    def test_organization(self, source_dir: str, target_dir: str, test_files: List[str]) -> Dict[str, Any]:
        """
        Test book organization functionality.
        
        Args:
            source_dir: Directory containing test files
            target_dir: Target directory for organized files
            test_files: List of test files to organize
            
        Returns:
            Test result dictionary
        """
        logger.info(f"Testing book organization on {len(test_files)} files")
        
        result = {
            "total_files": len(test_files),
            "organized_files": 0,
            "errors": 0,
            "time_taken": None,
            "details": []
        }
        
        try:
            # Create temporary directories for testing
            with tempfile.TemporaryDirectory() as temp_source_dir, tempfile.TemporaryDirectory() as temp_target_dir:
                # Copy test files to temporary source directory
                for file_path in test_files:
                    shutil.copy2(file_path, temp_source_dir)
                
                # Run book organizer
                start_time = time.time()
                total, organized, errors = self.book_organizer.organize_directory(
                    temp_source_dir,
                    temp_target_dir,
                    rename_files=True,
                    recursive=False,
                    dry_run=False
                )
                elapsed_time = time.time() - start_time
                
                result["organized_files"] = organized
                result["errors"] = errors
                result["time_taken"] = elapsed_time
                
                # Get details for organized directory structure
                for root, dirs, files in os.walk(temp_target_dir):
                    for file in files:
                        rel_path = os.path.relpath(os.path.join(root, file), temp_target_dir)
                        result["details"].append({
                            "file": file,
                            "path": rel_path
                        })
                
                logger.info(f"Organized {organized} out of {total} files with {errors} errors in {elapsed_time:.2f}s")
            
            # Add to results
            self.results["organization"].append(result)
            
        except Exception as e:
            logger.error(f"Error in organization test: {e}")
            result["error"] = str(e)
            self.results["organization"].append(result)
        
        return result
    
    def run_all_tests(self, test_dir: str):
        """
        Run all book module tests.
        
        Args:
            test_dir: Directory containing test files
        """
        logger.info(f"Running all book module tests on files in: {test_dir}")
        
        # Verify test directory exists
        if not os.path.isdir(test_dir):
            logger.error(f"Test directory not found: {test_dir}")
            return
        
        # Find book files in test directory
        book_extensions = ['.epub', '.mobi', '.azw', '.azw3', '.pdf', '.txt']
        test_files = []
        
        for root, _, files in os.walk(test_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in book_extensions):
                    test_files.append(os.path.join(root, file))
        
        logger.info(f"Found {len(test_files)} book files for testing")
        
        if not test_files:
            logger.warning(f"No book files found in {test_dir}")
            return
        
        # Limit to first 5 files for testing
        test_files = test_files[:5]
        
        # Run tests on each file
        for file_path in test_files:
            logger.info(f"Testing file: {os.path.basename(file_path)}")
            
            # Test metadata extraction
            metadata_result = self.test_metadata_extraction(file_path)
            
            # If metadata extraction succeeded, test filename generation
            if metadata_result["success"]:
                self.test_filename_generation(metadata_result["metadata"])
        
        # Test renaming and organization with all files
        if test_files:
            self.test_renaming(test_dir, test_files)
            self.test_organization(test_dir, os.path.join(test_dir, "organized"), test_files)
        
        # Print results
        self.print_results()
    
    def print_results(self):
        """Print test results in a readable format."""
        print("\n" + "="*60)
        print("BOOK MODULE TEST RESULTS")
        print("="*60)
        
        # Metadata extraction results
        print("\n1. METADATA EXTRACTION TESTS")
        print("-" * 40)
        
        for i, result in enumerate(self.results["metadata_extraction"], 1):
            if result["success"]:
                print(f"✅ Test {i}: {result['file']} - Success ({result['time_taken']:.2f}s)")
                if result["metadata"]:
                    # Print a few key metadata fields
                    metadata = result["metadata"]
                    print(f"   Title: {metadata.get('title', 'Unknown')}")
                    print(f"   Author: {metadata.get('author', 'Unknown')}")
                    print(f"   Year: {metadata.get('year', 'Unknown')}")
            else:
                print(f"❌ Test {i}: {result['file']} - Failed")
                if result["error"]:
                    print(f"   Error: {result['error']}")
        
        # Filename generation results
        print("\n2. FILENAME GENERATION TESTS")
        print("-" * 40)
        
        for i, result in enumerate(self.results["filename_generation"], 1):
            if result["success"]:
                print(f"✅ Test {i}: Success")
                print(f"   Standard: {result['standard_filename']}")
                print(f"   Smart: {result['smart_filename']}")
            else:
                print(f"❌ Test {i}: Failed")
                if result["error"]:
                    print(f"   Error: {result['error']}")
        
        # Renaming results
        print("\n3. BOOK RENAMING TESTS")
        print("-" * 40)
        
        for i, result in enumerate(self.results["renaming"], 1):
            success_rate = result["renamed_files"] / result["total_files"] if result["total_files"] > 0 else 0
            if success_rate > 0.5:
                print(f"✅ Test {i}: {result['renamed_files']}/{result['total_files']} files renamed ({result['time_taken']:.2f}s)")
            else:
                print(f"❌ Test {i}: Only {result['renamed_files']}/{result['total_files']} files renamed ({result['time_taken']:.2f}s)")
            
            if result.get("error"):
                print(f"   Error: {result['error']}")
        
        # Organization results
        print("\n4. BOOK ORGANIZATION TESTS")
        print("-" * 40)
        
        for i, result in enumerate(self.results["organization"], 1):
            success_rate = result["organized_files"] / result["total_files"] if result["total_files"] > 0 else 0
            if success_rate > 0.5:
                print(f"✅ Test {i}: {result['organized_files']}/{result['total_files']} files organized ({result['time_taken']:.2f}s)")
                if result["details"]:
                    print("   Directory structure:")
                    for detail in result["details"][:5]:  # Show first 5 for brevity
                        print(f"   - {detail['path']}")
                    if len(result["details"]) > 5:
                        print(f"     ... and {len(result['details']) - 5} more files")
            else:
                print(f"❌ Test {i}: Only {result['organized_files']}/{result['total_files']} files organized ({result['time_taken']:.2f}s)")
            
            if result.get("error"):
                print(f"   Error: {result['error']}")
        
        # Overall summary
        print("\n" + "="*60)
        
        metadata_success = sum(1 for r in self.results["metadata_extraction"] if r["success"])
        filename_success = sum(1 for r in self.results["filename_generation"] if r["success"])
        renaming_success = all(r["renamed_files"] > 0 for r in self.results["renaming"])
        organization_success = all(r["organized_files"] > 0 for r in self.results["organization"])
        
        all_success = (metadata_success > 0 and 
                      filename_success > 0 and 
                      renaming_success and 
                      organization_success)
        
        if all_success:
            print("✅ All book module tests passed successfully!")
        else:
            print("❌ Some book module tests failed.")
        
        # Print component status
        print(f"Metadata Extraction: {'✅' if metadata_success > 0 else '❌'} ({metadata_success}/{len(self.results['metadata_extraction'])} successful)")
        print(f"Filename Generation: {'✅' if filename_success > 0 else '❌'} ({filename_success}/{len(self.results['filename_generation'])} successful)")
        print(f"Book Renaming: {'✅' if renaming_success else '❌'}")
        print(f"Book Organization: {'✅' if organization_success else '❌'}")
        
        print("="*60 + "\n")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test book module functionality")
    parser.add_argument("--config", help="Path to custom config file")
    parser.add_argument("--dir", required=True, help="Directory containing test book files")
    
    args = parser.parse_args()
    
    try:
        # Check if book modules are available
        if not HAS_BOOK_MODULES:
            print("❌ Book modules not found. Make sure they are properly installed.")
            return 1
        
        # Initialize and run tests
        book_test = BookModuleTest(args.config)
        book_test.run_all_tests(args.dir)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in book module test: {e}", exc_info=True)
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
