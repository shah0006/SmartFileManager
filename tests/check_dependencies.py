#!/usr/bin/env python3
"""
Dependency Checker for SmartFileManager

This script checks if all required and optional dependencies
are installed and working properly, with special focus on
testing LLM integration.
"""

import sys
import os
import importlib
import platform
import subprocess
import time
import logging
from typing import Dict, List, Tuple, Any, Optional

# Add parent directory to path so we can import from the project
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("dependency-checker")

# Define required and optional dependencies
CORE_DEPENDENCIES = [
    "tqdm",        # Progress bars
    "requests",    # API calls
    "json",        # JSON handling
    "logging",     # Logging
    "pathlib",     # Path manipulation
    "datetime",    # Date handling
    "re",          # Regular expressions
    "concurrent.futures",  # Async processing
]

# Domain-specific dependencies
PDF_DEPENDENCIES = [
    "fitz",        # PyMuPDF for PDF handling
    "pdfminer",    # Alternative PDF handling
    "pdfplumber",  # Another PDF library
]

DOCUMENT_DEPENDENCIES = [
    "docx",        # Python-docx for Word documents
    "zipfile",     # For EPUB/ZIP handling
    "xml.etree.ElementTree",  # XML parsing (for EPUB)
]

IMAGE_DEPENDENCIES = [
    "PIL",         # Pillow for image handling
    "pytesseract", # OCR capabilities
]

MEDICAL_DEPENDENCIES = [
    "pydicom",     # DICOM medical image handling
]

WEB_DEPENDENCIES = [
    "bs4",         # BeautifulSoup for HTML parsing
]

LLM_DEPENDENCIES = [
    "transformers", # HuggingFace Transformers
    "torch",        # PyTorch for Transformers
    "ollama",       # Python library for Ollama
]

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def check_import(module_name: str) -> Tuple[bool, str, Optional[str]]:
    """
    Check if a module can be imported.
    
    Args:
        module_name: Name of the module to import
        
    Returns:
        Tuple of (success, version, error message)
    """
    try:
        # Handle special cases
        if module_name == "fitz":
            import fitz
            return True, fitz.version, None
        elif module_name == "bs4":
            import bs4
            return True, bs4.__version__, None
        elif module_name == "PIL":
            import PIL
            return True, PIL.__version__, None
        else:
            # Try to import the module
            module = importlib.import_module(module_name)
            
            # Try to get version (not all modules have __version__)
            try:
                version = module.__version__
            except AttributeError:
                version = "Unknown"
            
            return True, version, None
    except ImportError as e:
        return False, "", str(e)
    except Exception as e:
        return False, "", str(e)

def check_command(command: List[str]) -> Tuple[bool, str, Optional[str]]:
    """
    Check if a command can be executed.
    
    Args:
        command: Command to execute as a list of strings
        
    Returns:
        Tuple of (success, output, error message)
    """
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode == 0:
            return True, result.stdout.strip(), None
        else:
            return False, "", result.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "", "Command timed out"
    except Exception as e:
        return False, "", str(e)

def check_ollama() -> Tuple[bool, str, Optional[str]]:
    """
    Check if Ollama is installed and running.
    
    Returns:
        Tuple of (success, version/status, error message)
    """
    # First check if the Ollama command is available
    success, output, error = check_command(["ollama", "version"])
    
    if not success:
        return False, "", "Ollama CLI not available: " + (error or "unknown error")
    
    # Then check if the Ollama server is running
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        
        if response.status_code == 200:
            return True, f"CLI: {output}, Server: Running", None
        else:
            return False, output, f"Ollama server returned status code {response.status_code}"
    except requests.exceptions.ConnectionError:
        return False, output, "Ollama server not running"
    except Exception as e:
        return False, output, str(e)

def test_llm_integration() -> Dict[str, Any]:
    """
    Test LLM integration by trying to load and use a local LLM.
    
    Returns:
        Dictionary with test results
    """
    results = {
        "transformers": {
            "available": False,
            "model_loaded": False,
            "inference_test": False,
            "error": None,
            "output": None,
            "time_taken": None
        },
        "ollama": {
            "available": False,
            "api_accessible": False,
            "inference_test": False,
            "error": None,
            "output": None,
            "time_taken": None
        }
    }
    
    # Test Transformers
    try:
        transformers_available, _, error = check_import("transformers")
        results["transformers"]["available"] = transformers_available
        
        if transformers_available:
            torch_available, _, _ = check_import("torch")
            
            if torch_available:
                # Try to load a small model and run inference
                import torch
                from transformers import AutoTokenizer, AutoModelForCausalLM
                
                start_time = time.time()
                
                # Use TinyLlama as it's relatively small
                model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
                
                try:
                    logger.info(f"Loading Transformers model: {model_name}")
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    results["transformers"]["model_loaded"] = True
                    
                    # Use CPU for the test
                    model = AutoModelForCausalLM.from_pretrained(model_name)
                    
                    # Run a simple inference
                    prompt = "Extract the title and author from this text: 'Introduction to Machine Learning by John Smith'"
                    inputs = tokenizer(prompt, return_tensors="pt")
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            inputs["input_ids"],
                            max_new_tokens=50,
                            temperature=0.2,
                            do_sample=True
                        )
                    
                    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                    results["transformers"]["inference_test"] = True
                    results["transformers"]["output"] = response
                    results["transformers"]["time_taken"] = time.time() - start_time
                    
                except Exception as e:
                    results["transformers"]["error"] = str(e)
            else:
                results["transformers"]["error"] = "PyTorch not available"
        else:
            results["transformers"]["error"] = error
    except Exception as e:
        results["transformers"]["error"] = str(e)
    
    # Test Ollama
    try:
        ollama_running, status, error = check_ollama()
        results["ollama"]["available"] = ollama_running
        
        if ollama_running:
            try:
                import requests
                
                # Check if API is accessible
                response = requests.get("http://localhost:11434/api/tags", timeout=2)
                if response.status_code == 200:
                    results["ollama"]["api_accessible"] = True
                    
                    # Run a simple inference
                    start_time = time.time()
                    
                    prompt = "Extract the title and author from this text: 'Introduction to Machine Learning by John Smith'"
                    payload = {
                        "model": "mistral:7b",  # Use a common model
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.2,
                            "num_predict": 50
                        }
                    }
                    
                    logger.info("Testing Ollama inference with model: mistral:7b")
                    inference_response = requests.post(
                        "http://localhost:11434/api/generate",
                        json=payload,
                        timeout=30  # Longer timeout for inference
                    )
                    
                    if inference_response.status_code == 200:
                        results["ollama"]["inference_test"] = True
                        results["ollama"]["output"] = inference_response.json().get("response", "")
                        results["ollama"]["time_taken"] = time.time() - start_time
                    else:
                        results["ollama"]["error"] = f"API error: {inference_response.status_code} - {inference_response.text}"
                else:
                    results["ollama"]["error"] = f"API returned status code {response.status_code}"
            except requests.exceptions.ConnectionError:
                results["ollama"]["error"] = "Ollama API not accessible"
            except Exception as e:
                results["ollama"]["error"] = str(e)
        else:
            results["ollama"]["error"] = error
    except Exception as e:
        results["ollama"]["error"] = str(e)
    
    return results

def check_all_dependencies() -> Dict[str, Dict[str, Any]]:
    """
    Check all dependencies.
    
    Returns:
        Dictionary with results for all dependencies
    """
    results = {}
    
    # Check system info
    system_info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    results["system"] = system_info
    
    # Check core dependencies
    core_results = {}
    for dep in CORE_DEPENDENCIES:
        success, version, error = check_import(dep)
        core_results[dep] = {
            "installed": success,
            "version": version,
            "error": error
        }
    results["core"] = core_results
    
    # Check PDF dependencies
    pdf_results = {}
    for dep in PDF_DEPENDENCIES:
        success, version, error = check_import(dep)
        pdf_results[dep] = {
            "installed": success,
            "version": version,
            "error": error
        }
    results["pdf"] = pdf_results
    
    # Check document dependencies
    doc_results = {}
    for dep in DOCUMENT_DEPENDENCIES:
        success, version, error = check_import(dep)
        doc_results[dep] = {
            "installed": success,
            "version": version,
            "error": error
        }
    results["document"] = doc_results
    
    # Check image dependencies
    image_results = {}
    for dep in IMAGE_DEPENDENCIES:
        success, version, error = check_import(dep)
        image_results[dep] = {
            "installed": success,
            "version": version,
            "error": error
        }
    results["image"] = image_results
    
    # Check medical dependencies
    medical_results = {}
    for dep in MEDICAL_DEPENDENCIES:
        success, version, error = check_import(dep)
        medical_results[dep] = {
            "installed": success,
            "version": version,
            "error": error
        }
    results["medical"] = medical_results
    
    # Check web dependencies
    web_results = {}
    for dep in WEB_DEPENDENCIES:
        success, version, error = check_import(dep)
        web_results[dep] = {
            "installed": success,
            "version": version,
            "error": error
        }
    results["web"] = web_results
    
    # Check LLM dependencies
    llm_results = {}
    for dep in LLM_DEPENDENCIES:
        success, version, error = check_import(dep)
        llm_results[dep] = {
            "installed": success,
            "version": version,
            "error": error
        }
    results["llm"] = llm_results
    
    # Test LLM integration
    results["llm_integration"] = test_llm_integration()
    
    return results

def print_results(results: Dict[str, Dict[str, Any]]):
    """
    Print dependency check results in a readable format.
    
    Args:
        results: Results dictionary from check_all_dependencies
    """
    def print_section(title: str, section_results: Dict[str, Any]):
        print(f"\n{Colors.BOLD}{Colors.UNDERLINE}{title}{Colors.END}")
        for name, data in section_results.items():
            if isinstance(data, dict) and "installed" in data:
                if data["installed"]:
                    status = f"{Colors.GREEN}✓{Colors.END}"
                    version_info = f"v{data['version']}" if data["version"] else ""
                else:
                    status = f"{Colors.RED}✗{Colors.END}"
                    version_info = f"{Colors.RED}{data['error']}{Colors.END}" if data["error"] else ""
                
                print(f"  {status} {name:<20} {version_info}")
            else:
                print(f"  {name}: {data}")
    
    print(f"\n{Colors.BOLD}{Colors.BLUE}SmartFileManager Dependency Check Results{Colors.END}")
    print(f"{Colors.BOLD}System Information:{Colors.END}")
    print(f"  Python: {results['system']['python_version'].split()[0]}")
    print(f"  Platform: {results['system']['platform']}")
    
    # Print regular dependencies
    print_section("Core Dependencies", results["core"])
    print_section("PDF Processing", results["pdf"])
    print_section("Document Processing", results["document"])
    print_section("Image Processing", results["image"])
    print_section("Medical File Processing", results["medical"])
    print_section("Web Content Processing", results["web"])
    print_section("LLM Libraries", results["llm"])
    
    # Print LLM integration test results
    print(f"\n{Colors.BOLD}{Colors.UNDERLINE}LLM Integration Tests{Colors.END}")
    
    # Transformers results
    transformers = results["llm_integration"]["transformers"]
    if transformers["available"]:
        if transformers["inference_test"]:
            status = f"{Colors.GREEN}PASSED{Colors.END}"
            details = f"Model loaded and inference successful (took {transformers['time_taken']:.2f}s)"
        elif transformers["model_loaded"]:
            status = f"{Colors.YELLOW}PARTIAL{Colors.END}"
            details = f"Model loaded but inference failed: {transformers['error']}"
        else:
            status = f"{Colors.RED}FAILED{Colors.END}"
            details = f"Failed to load model: {transformers['error']}"
    else:
        status = f"{Colors.YELLOW}SKIPPED{Colors.END}"
        details = "Transformers not available"
    
    print(f"  HuggingFace Transformers: {status}")
    print(f"    {details}")
    if transformers["output"]:
        print(f"    Sample output: {Colors.BLUE}{transformers['output'][:100]}...{Colors.END}")
    
    # Ollama results
    ollama = results["llm_integration"]["ollama"]
    if ollama["available"]:
        if ollama["inference_test"]:
            status = f"{Colors.GREEN}PASSED{Colors.END}"
            details = f"API accessible and inference successful (took {ollama['time_taken']:.2f}s)"
        elif ollama["api_accessible"]:
            status = f"{Colors.YELLOW}PARTIAL{Colors.END}"
            details = f"API accessible but inference failed: {ollama['error']}"
        else:
            status = f"{Colors.RED}FAILED{Colors.END}"
            details = f"API not accessible: {ollama['error']}"
    else:
        status = f"{Colors.YELLOW}SKIPPED{Colors.END}"
        details = "Ollama not available or not running"
    
    print(f"  Ollama API: {status}")
    print(f"    {details}")
    if ollama["output"]:
        print(f"    Sample output: {Colors.BLUE}{ollama['output'][:100]}...{Colors.END}")
    
    # Print overall LLM availability
    if transformers["inference_test"] or ollama["inference_test"]:
        print(f"\n{Colors.GREEN}{Colors.BOLD}LLM Integration: AVAILABLE{Colors.END}")
        
        if transformers["inference_test"]:
            print(f"  {Colors.GREEN}✓{Colors.END} Primary: Transformers")
        elif ollama["inference_test"]:
            print(f"  {Colors.GREEN}✓{Colors.END} Primary: Ollama")
        
        if transformers["inference_test"] and ollama["inference_test"]:
            print(f"  {Colors.GREEN}✓{Colors.END} Both LLM providers are available")
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}LLM Integration: NOT AVAILABLE{Colors.END}")
        print("  To use LLM features, install and configure either:")
        print("    - HuggingFace Transformers (pip install transformers torch)")
        print("    - Ollama (https://ollama.ai/)")

def main():
    """Main function to run all checks."""
    logger.info("Starting dependency check")
    
    try:
        results = check_all_dependencies()
        print_results(results)
        
        # Determine if we have all critical dependencies
        core_missing = [dep for dep in CORE_DEPENDENCIES 
                        if not results["core"].get(dep, {}).get("installed", False)]
        
        # Check if we have at least one PDF library
        pdf_available = any(results["pdf"].get(dep, {}).get("installed", False) 
                           for dep in PDF_DEPENDENCIES)
        
        # Check LLM availability
        llm_available = (results["llm_integration"]["transformers"]["inference_test"] or 
                        results["llm_integration"]["ollama"]["inference_test"])
        
        print("\n" + "="*50)
        if not core_missing and pdf_available:
            print(f"{Colors.GREEN}{Colors.BOLD}READY FOR TESTING{Colors.END}")
            print("All critical dependencies are available.")
            
            if not llm_available:
                print(f"{Colors.YELLOW}WARNING:{Colors.END} LLM integration not available.")
                print("Some features requiring content analysis will be limited.")
        else:
            print(f"{Colors.YELLOW}{Colors.BOLD}DEPENDENCY ISSUES DETECTED{Colors.END}")
            
            if core_missing:
                print(f"Missing core dependencies: {', '.join(core_missing)}")
            
            if not pdf_available:
                print("No PDF processing library available (PyMuPDF, pdfminer, or pdfplumber required)")
        
        print("="*50)
        
    except Exception as e:
        logger.error(f"Error during dependency check: {e}", exc_info=True)
        print(f"{Colors.RED}Error during dependency check: {e}{Colors.END}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
