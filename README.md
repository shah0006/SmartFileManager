# SmartFileManager

A comprehensive, intelligent file management system that organizes files by content domain, extracts metadata using local LLMs, and ensures safe file operations with progress tracking.

## Overview

SmartFileManager is a consolidated solution that combines and enhances the functionality of the Smart File Organizer and RenameBooksInFolder projects. It provides specialized handling for different content domains:

- **Academic Papers**: Extracts metadata from PDFs, renames according to academic conventions, integrates with Zotero
- **Books**: Processes EPUB, MOBI, PDF and other book formats with metadata-based organization
- **Medical Files**: Handles medical documents with privacy-aware organization
- **General Files**: Manages common file types with content-based organization

## Features

- **Content-Domain Organization**: Files are organized based on their content category
- **Local LLM Integration**: Uses local language models (Transformers or Ollama) for metadata extraction and content analysis
- **Safe File Operations**: Moves files to a "to-delete" folder instead of permanent deletion
- **Progress Tracking**: Visual feedback with progress bars during operations
- **Zotero Integration**: Compares local files with Zotero library for consistent naming and duplicate detection
- **Privacy Controls**: Enhanced privacy features for sensitive documents
- **Duplicate Detection**: Identifies duplicate files based on content hash
- **Customizable Configuration**: Flexible settings for each module

## Project Structure

```
SmartFileManager/
├── core/                  # Core functionality
│   ├── file_operations.py # Safe file operations
│   └── metadata_extractor.py # Base metadata extraction
├── modules/               # Domain-specific modules
│   ├── academic/          # Academic papers handling
│   │   ├── pdf_metadata.py
│   │   ├── pdf_renamer.py
│   │   └── zotero_integration.py
│   ├── books/             # Book files handling
│   │   ├── book_metadata.py
│   │   ├── book_renamer.py
│   │   └── book_organizer.py
│   ├── medical/           # Medical files handling
│   │   ├── medical_metadata.py
│   │   └── medical_organizer.py
│   └── general/           # General files handling
│       ├── file_analyzer.py
│       └── file_organizer.py
├── ui/                    # User interfaces
│   └── cli.py             # Command-line interface
├── utils/                 # Utility modules
│   └── llm_handler.py     # LLM integration
├── config/                # Configuration
│   └── config.py          # Configuration management
├── logs/                  # Log files
├── data/                  # Data storage
├── to_delete/             # Safe deletion storage
└── main.py                # Main entry point
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   ```

2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. (Optional) Install local LLM dependencies:
   ```
   # For Transformers support
   pip install transformers torch
   
   # For Ollama support (requires Ollama server running)
   # See https://ollama.ai/ for installation instructions
   ```

## Usage

### Command Line Interface

The main script provides access to all functionality:

```
python main.py <command> [options]
```

Available commands:

- **Book Management**:
  ```
  # Rename books based on metadata
  python main.py rename-books <directory> [--recursive]
  
  # Organize books into directories
  python main.py organize-books <source> <target> [--rename] [--by-author] [--by-year]
  ```

- **Academic Paper Management**:
  ```
  # Rename academic PDFs
  python main.py rename-pdfs <directory> [--recursive] [--use-llm]
  
  # Compare with Zotero library
  python main.py compare-zotero <directory> <zotero_dir> [--report]
  ```

- **Medical File Management**:
  ```
  # Organize medical files
  python main.py organize-medical <source> <target> [--rename] [--privacy=high|medium|low]
  ```

- **General File Management**:
  ```
  # Organize general files
  python main.py organize <source> <target> [--recursive] [--by-type] [--by-date] [--content]
  
  # Find duplicate files
  python main.py find-duplicates <directory> [--recursive] [--report]
  
  # Clean empty directories
  python main.py clean-empty <directory>
  ```

### Configuration

The default configuration is stored in `config/settings.json`. You can modify this file or specify a custom configuration file:

```
python main.py --config custom_config.json
```

## Dependencies

- Python 3.7+
- PyMuPDF/fitz (for PDF processing)
- tqdm (for progress bars)
- requests (for API communication)
- Optional:
  - transformers, torch (for HuggingFace Transformers LLM support)
  - ollama (for Ollama LLM support)
  - python-docx (for Word document processing)
  - pydicom (for DICOM medical image processing)
  - pytesseract (for OCR)
  - beautifulsoup4 (for HTML parsing)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
