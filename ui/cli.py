"""
Command-Line Interface for SmartFileManager.

This module provides a unified CLI for accessing all the 
functionality of SmartFileManager.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import concurrent.futures
from typing import Dict, List, Any, Optional, Tuple

# Core modules
from core.file_operations import FileOperations

# Domain-specific modules
try:
    from modules.books.book_renamer import BookRenamer
    from modules.books.book_organizer import BookOrganizer
    HAS_BOOK_MODULES = True
except ImportError:
    HAS_BOOK_MODULES = False

try:
    from modules.academic.pdf_renamer import PDFRenamer
    from modules.academic.pdf_metadata import PDFMetadataExtractor
    from modules.academic.zotero_integration import ZoteroComparison
    HAS_ACADEMIC_MODULES = True
except ImportError:
    HAS_ACADEMIC_MODULES = False

try:
    from modules.medical.medical_organizer import MedicalFileOrganizer
    HAS_MEDICAL_MODULES = True
except ImportError:
    HAS_MEDICAL_MODULES = False

try:
    from modules.general.file_organizer import SmartFileOrganizer
    HAS_GENERAL_MODULES = True
except ImportError:
    HAS_GENERAL_MODULES = False

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Define a simple fallback
    class tqdm:
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable
            self.total = len(iterable) if iterable is not None else 0
            self.n = 0
            self.desc = kwargs.get('desc', '')
            
        def __iter__(self):
            for obj in self.iterable:
                self.n += 1
                if self.n % 10 == 0 or self.n == self.total:
                    print(f"{self.desc}: {self.n}/{self.total}", file=sys.stderr)
                yield obj
                
        def update(self, n=1):
            self.n += n
            
        def close(self):
            pass

# LLM Handler (import if available)
try:
    from utils.llm_handler import LLMHandler
    HAS_LLM = True
except ImportError:
    HAS_LLM = False

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), '..', 'logs', 'smartfilemanager.log'), mode='a')
    ]
)

class SmartFileManagerCLI:
    """Main CLI class for SmartFileManager."""
    
    def __init__(self):
        """Initialize the CLI."""
        self.file_ops = FileOperations()
        self.llm_handler = self._initialize_llm()
        
        # Create a parser
        self.parser = argparse.ArgumentParser(
            description="SmartFileManager - Intelligent file organization and management",
            formatter_class=argparse.RawTextHelpFormatter
        )
        
        self._setup_parsers()
    
    def _initialize_llm(self) -> Optional[Any]:
        """Initialize the LLM handler if available."""
        if not HAS_LLM:
            return None
            
        try:
            # Try to initialize the LLM handler
            llm_handler = LLMHandler()
            return llm_handler
        except Exception as e:
            logger.warning(f"Error initializing LLM handler: {e}")
            return None
    
    def _setup_parsers(self):
        """Set up command-line argument parsers."""
        # Add global arguments
        self.parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
        self.parser.add_argument("--dry-run", action="store_true", help="Show actions without performing them")
        
        # Create subparsers for different commands
        subparsers = self.parser.add_subparsers(dest="command", help="Command to execute")
        
        # Books commands
        if HAS_BOOK_MODULES:
            self._setup_books_parser(subparsers)
        
        # Academic commands
        if HAS_ACADEMIC_MODULES:
            self._setup_academic_parser(subparsers)
        
        # Medical commands
        if HAS_MEDICAL_MODULES:
            self._setup_medical_parser(subparsers)
        
        # General commands
        if HAS_GENERAL_MODULES:
            self._setup_general_parser(subparsers)
        
        # Common commands
        self._setup_common_parser(subparsers)
    
    def _setup_books_parser(self, subparsers):
        """Set up parser for book-related commands."""
        # Book renaming
        book_rename_parser = subparsers.add_parser("rename-books", help="Rename book files based on metadata")
        book_rename_parser.add_argument("directory", help="Directory containing book files")
        book_rename_parser.add_argument("--recursive", "-r", action="store_true", help="Process subdirectories")
        book_rename_parser.add_argument("--report", action="store_true", help="Generate a detailed report")
        
        # Book organization
        book_organize_parser = subparsers.add_parser("organize-books", help="Organize book files into directories")
        book_organize_parser.add_argument("source", help="Source directory containing book files")
        book_organize_parser.add_argument("target", help="Target directory for organized books")
        book_organize_parser.add_argument("--recursive", "-r", action="store_true", help="Process subdirectories")
        book_organize_parser.add_argument("--rename", action="store_true", help="Rename files during organization")
        book_organize_parser.add_argument("--by-author", action="store_true", help="Organize by author")
        book_organize_parser.add_argument("--by-year", action="store_true", help="Organize by year")
    
    def _setup_academic_parser(self, subparsers):
        """Set up parser for academic-related commands."""
        # PDF renaming
        pdf_rename_parser = subparsers.add_parser("rename-pdfs", help="Rename academic PDF files")
        pdf_rename_parser.add_argument("directory", help="Directory containing PDF files")
        pdf_rename_parser.add_argument("--recursive", "-r", action="store_true", help="Process subdirectories")
        pdf_rename_parser.add_argument("--use-llm", action="store_true", help="Use LLM for metadata extraction")
        
        # Zotero comparison
        zotero_parser = subparsers.add_parser("compare-zotero", help="Compare files with Zotero library")
        zotero_parser.add_argument("directory", help="Directory containing academic files")
        zotero_parser.add_argument("zotero_dir", help="Zotero storage directory")
        zotero_parser.add_argument("--report", action="store_true", help="Generate a detailed report")
    
    def _setup_medical_parser(self, subparsers):
        """Set up parser for medical-related commands."""
        # Medical file organization
        medical_parser = subparsers.add_parser("organize-medical", help="Organize medical files")
        medical_parser.add_argument("source", help="Source directory containing medical files")
        medical_parser.add_argument("target", help="Target directory for organized medical files")
        medical_parser.add_argument("--recursive", "-r", action="store_true", help="Process subdirectories")
        medical_parser.add_argument("--rename", action="store_true", help="Rename files during organization")
        medical_parser.add_argument("--privacy", choices=["low", "medium", "high"], default="high", 
                               help="Privacy level for medical file organization")
    
    def _setup_general_parser(self, subparsers):
        """Set up parser for general file organization commands."""
        # General file organization
        organize_parser = subparsers.add_parser("organize", help="Organize general files")
        organize_parser.add_argument("source", help="Source directory containing files")
        organize_parser.add_argument("target", help="Target directory for organized files")
        organize_parser.add_argument("--recursive", "-r", action="store_true", help="Process subdirectories")
        organize_parser.add_argument("--by-type", action="store_true", help="Organize by file type")
        organize_parser.add_argument("--by-date", action="store_true", help="Organize by date")
        organize_parser.add_argument("--content", action="store_true", help="Use content-based organization (requires LLM)")
    
    def _setup_common_parser(self, subparsers):
        """Set up parser for common commands."""
        # Find duplicates
        duplicates_parser = subparsers.add_parser("find-duplicates", help="Find duplicate files")
        duplicates_parser.add_argument("directory", help="Directory to scan for duplicates")
        duplicates_parser.add_argument("--recursive", "-r", action="store_true", help="Process subdirectories")
        duplicates_parser.add_argument("--report", action="store_true", help="Generate a detailed report")
        
        # Clean empty directories
        clean_parser = subparsers.add_parser("clean-empty", help="Remove empty directories")
        clean_parser.add_argument("directory", help="Directory to clean")
    
    def run(self, args=None):
        """
        Run the CLI with the provided arguments.
        
        Args:
            args: Command-line arguments (if None, sys.argv is used)
        """
        # Parse arguments
        args = self.parser.parse_args(args)
        
        # Set up logging based on verbosity
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Execute the requested command
        if args.command == "rename-books" and HAS_BOOK_MODULES:
            self._rename_books(args)
        elif args.command == "organize-books" and HAS_BOOK_MODULES:
            self._organize_books(args)
        elif args.command == "rename-pdfs" and HAS_ACADEMIC_MODULES:
            self._rename_pdfs(args)
        elif args.command == "compare-zotero" and HAS_ACADEMIC_MODULES:
            self._compare_zotero(args)
        elif args.command == "organize-medical" and HAS_MEDICAL_MODULES:
            self._organize_medical(args)
        elif args.command == "organize" and HAS_GENERAL_MODULES:
            self._organize_files(args)
        elif args.command == "find-duplicates":
            self._find_duplicates(args)
        elif args.command == "clean-empty":
            self._clean_empty_dirs(args)
        else:
            self.parser.print_help()
    
    def _rename_books(self, args):
        """Rename book files."""
        logger.info(f"Renaming books in {args.directory}")
        
        # Initialize the book renamer
        book_renamer = BookRenamer(llm_handler=self.llm_handler)
        
        # Process the directory
        total, renamed, errors = book_renamer.process_directory(
            args.directory,
            recursive=args.recursive,
            rename_immediately=not args.dry_run
        )
        
        logger.info(f"Processed {total} books, renamed {renamed}, encountered {errors} errors")
    
    def _organize_books(self, args):
        """Organize book files."""
        logger.info(f"Organizing books from {args.source} to {args.target}")
        
        # Initialize the book organizer
        book_organizer = BookOrganizer(llm_handler=self.llm_handler)
        
        # Configure organizer
        if args.by_author:
            book_organizer.organize_by_author = True
        if args.by_year:
            book_organizer.organize_by_year = True
        
        # Process the directory
        total, organized, errors = book_organizer.organize_directory(
            args.source,
            args.target,
            rename_files=args.rename,
            recursive=args.recursive,
            dry_run=args.dry_run
        )
        
        logger.info(f"Processed {total} books, organized {organized}, encountered {errors} errors")
    
    def _rename_pdfs(self, args):
        """Rename academic PDF files."""
        logger.info(f"Renaming PDFs in {args.directory}")
        
        # Initialize the PDF renamer
        pdf_renamer = PDFRenamer(use_llm=args.use_llm, llm_handler=self.llm_handler)
        
        # Process the directory
        results = pdf_renamer.rename_pdfs_in_directory(
            args.directory,
            recursive=args.recursive,
            dry_run=args.dry_run
        )
        
        logger.info(f"PDF renaming results: {results}")
    
    def _compare_zotero(self, args):
        """Compare files with Zotero library."""
        logger.info(f"Comparing files in {args.directory} with Zotero library in {args.zotero_dir}")
        
        # Initialize Zotero comparison
        zotero_comp = ZoteroComparison()
        
        # Run comparison
        results = zotero_comp.compare_directories(
            args.directory,
            args.zotero_dir,
            generate_report=args.report
        )
        
        logger.info(f"Zotero comparison results: {results}")
    
    def _organize_medical(self, args):
        """Organize medical files."""
        logger.info(f"Organizing medical files from {args.source} to {args.target}")
        
        # Initialize medical organizer with privacy settings
        config = {"privacy_level": args.privacy}
        medical_organizer = MedicalFileOrganizer(llm_handler=self.llm_handler, config=config)
        
        # Process the directory
        total, organized, errors = medical_organizer.organize_directory(
            args.source,
            args.target,
            rename_files=args.rename,
            recursive=args.recursive,
            dry_run=args.dry_run
        )
        
        logger.info(f"Processed {total} medical files, organized {organized}, encountered {errors} errors")
    
    def _organize_files(self, args):
        """Organize general files."""
        logger.info(f"Organizing files from {args.source} to {args.target}")
        
        # Initialize file organizer with config
        config = {
            "organize_by_type": args.by_type,
            "organize_by_date": args.by_date,
            "organize_by_content": args.content
        }
        file_organizer = SmartFileOrganizer(llm_handler=self.llm_handler, config=config)
        
        # Process the directory
        total, organized, errors = file_organizer.organize_directory(
            args.source,
            args.target,
            recursive=args.recursive,
            dry_run=args.dry_run
        )
        
        # Clean empty directories if requested
        if not args.dry_run and config.get("clean_empty_dirs", False):
            cleaned = file_organizer.clean_empty_directories(args.source)
            logger.info(f"Cleaned {cleaned} empty directories")
        
        logger.info(f"Processed {total} files, organized {organized}, encountered {errors} errors")
    
    def _find_duplicates(self, args):
        """Find duplicate files in a directory."""
        logger.info(f"Finding duplicates in {args.directory}")
        
        # Initialize file analyzer
        from modules.general.file_analyzer import FileAnalyzer
        file_analyzer = FileAnalyzer()
        
        # Get all files
        all_files = []
        if args.recursive:
            for root, _, files in os.walk(args.directory):
                for file in files:
                    all_files.append(os.path.join(root, file))
        else:
            for file in os.listdir(args.directory):
                if os.path.isfile(os.path.join(args.directory, file)):
                    all_files.append(os.path.join(args.directory, file))
        
        # Find duplicates
        duplicates = file_analyzer.identify_duplicates(all_files)
        
        # Report results
        if duplicates:
            logger.info(f"Found {len(duplicates)} duplicate file groups")
            
            # Generate a report if requested
            if args.report:
                import datetime
                report_file = f"duplicates_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write(f"Duplicate Files Report for {args.directory}\n")
                    f.write(f"Generated on: {datetime.datetime.now()}\n\n")
                    
                    for hash_value, file_list in duplicates.items():
                        f.write(f"Hash: {hash_value}\n")
                        for file_path in file_list:
                            f.write(f"  - {file_path}\n")
                        f.write("\n")
                
                logger.info(f"Duplicate report generated: {report_file}")
            else:
                # Simple console output
                for hash_value, file_list in duplicates.items():
                    print(f"Duplicate files (hash: {hash_value[:8]}...):")
                    for file_path in file_list:
                        print(f"  - {file_path}")
                    print()
        else:
            logger.info("No duplicate files found")
    
    def _clean_empty_dirs(self, args):
        """Clean empty directories."""
        logger.info(f"Cleaning empty directories in {args.directory}")
        
        # Count empty directories
        count = 0
        
        # Walk bottom-up
        for root, dirs, files in os.walk(args.directory, topdown=False):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                
                # Check if directory is empty
                if not os.listdir(dir_path):
                    if not args.dry_run:
                        os.rmdir(dir_path)
                    count += 1
                    logger.info(f"{'Would remove' if args.dry_run else 'Removed'} empty directory: {dir_path}")
        
        logger.info(f"{'Would clean' if args.dry_run else 'Cleaned'} {count} empty directories")


def main():
    """Main entry point for the CLI."""
    try:
        # Ensure logs directory exists
        log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Run the CLI
        cli = SmartFileManagerCLI()
        cli.run()
        
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
