#!/usr/bin/env python3
"""
LLM Handler Test Script for SmartFileManager

This script tests the LLM handler functionality specifically
in the context of metadata extraction tasks.
"""

import sys
import os
import time
import logging
import argparse
from typing import Dict, Any, Optional, List

# Add parent directory to path so we can import from the project
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import LLM handler
from utils.llm_handler import get_llm_handler
from config.config import get_config_manager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("llm-test")

# Test metadata extraction prompts
TEST_PROMPTS = [
    # Academic paper metadata extraction
    """
    Extract the following metadata from this academic paper text:
    Title, Authors, Year, Journal or Conference, DOI.
    
    Paper text:
    Deep Residual Learning for Image Recognition
    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Microsoft Research
    {kahe, v-xiangz, v-shren, jiansun}@microsoft.com
    
    Abstract
    Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth. On the ImageNet dataset we evaluate residual nets with a depth of up to 152 layers—8× deeper than VGG nets but still having lower complexity. An ensemble of these residual nets achieves 3.57% error on the ImageNet test set. This result won the 1st place on the ILSVRC 2015 classification task. We also present analysis on CIFAR-10 with 100 and 1000 layers.
    
    Introduction
    Deep convolutional neural networks have led to a series of breakthroughs for image classification [21, 50, 51]. Deep networks naturally integrate low/mid/high-level features [50] and classifiers in an end-to-end multi-layer fashion, and the "levels" of features can be enriched by the number of stacked layers (depth). Recent evidence [41, 42] reveals that network depth is of crucial importance, and the leading results [41, 42, 44, 47, 49] on the challenging ImageNet dataset [36] all exploit "very deep" [41] models.
    """,
    
    # Book metadata extraction
    """
    Extract the following metadata from this book information:
    Title, Author, Year, Publisher, ISBN.
    
    Book information:
    THE MIDNIGHT LIBRARY
    MATT HAIG
    VIKING
    An imprint of Penguin Random House LLC
    penguinrandomhouse.com
    
    First published in the United States of America by Viking, an imprint of
    Penguin Random House LLC, 2020
    Copyright © 2020 by Matt Haig
    
    ISBN 9780525559474 (hardcover)
    ISBN 9780525559481 (ebook)
    ISBN 9780655697077 (international edition)
    """,
    
    # Medical document metadata extraction
    """
    Extract the following metadata from this medical document:
    Document Type, Patient ID (use 'Anonymous' if present), Date, Physician, Facility.
    
    Medical document:
    RADIOLOGY REPORT
    
    Patient: John Smith
    Medical Record #: MRN12345
    Date of Service: 05/12/2023
    Referring Physician: Dr. Jane Williams
    Facility: Memorial Hospital
    
    EXAM: Chest X-ray, PA and Lateral
    
    CLINICAL INDICATION: Shortness of breath, rule out pneumonia
    
    TECHNIQUE: PA and lateral chest radiographs were obtained.
    
    FINDINGS:
    The lungs are clear without focal consolidation, effusion, or pneumothorax.
    The cardiomediastinal silhouette is normal in size and contour.
    The visualized bony structures show no acute abnormality.
    
    IMPRESSION:
    Normal chest radiograph.
    
    Electronically signed by:
    Robert Johnson, MD
    Radiologist
    05/12/2023 14:30
    """
]

def test_llm_handler(config_file: Optional[str] = None, provider: Optional[str] = None):
    """
    Test the LLM handler with sample metadata extraction tasks.
    
    Args:
        config_file: Optional path to custom config file
        provider: Optional LLM provider to use (overrides config)
    """
    # Initialize configuration
    config_manager = get_config_manager(config_file)
    llm_config = config_manager.get_config("llm")
    
    # Override provider if specified
    if provider:
        llm_config["provider"] = provider
    
    # Initialize LLM handler
    logger.info(f"Initializing LLM handler with provider: {llm_config.get('provider', 'auto')}")
    llm_handler = get_llm_handler(llm_config)
    
    if not llm_handler.is_available():
        logger.error("No LLM provider available. Please check your configuration and dependencies.")
        return False
    
    logger.info(f"LLM provider initialized: {llm_handler.provider}, model: {llm_handler.model}")
    
    # Test with prompts
    results = []
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        logger.info(f"Running test {i}/3: {prompt.splitlines()[0]}")
        
        try:
            start_time = time.time()
            response = llm_handler.get_response(prompt)
            elapsed_time = time.time() - start_time
            
            results.append({
                "test_id": i,
                "success": bool(response),
                "time_taken": elapsed_time,
                "response": response,
                "error": None
            })
            
            logger.info(f"Test {i} completed in {elapsed_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error in test {i}: {e}")
            results.append({
                "test_id": i,
                "success": False,
                "time_taken": None,
                "response": None,
                "error": str(e)
            })
    
    # Print results
    print("\n" + "="*60)
    print("LLM HANDLER TEST RESULTS")
    print("="*60)
    
    all_success = True
    for result in results:
        if result["success"]:
            print(f"\n✅ Test {result['test_id']} - Success ({result['time_taken']:.2f}s)")
            print(f"Response ({len(result['response'])} chars):")
            print("-" * 40)
            print(result["response"][:400] + ("..." if len(result["response"]) > 400 else ""))
            print("-" * 40)
        else:
            all_success = False
            print(f"\n❌ Test {result['test_id']} - Failed")
            print(f"Error: {result['error']}")
    
    print("\n" + "="*60)
    if all_success:
        print("✅ All tests passed successfully!")
        print(f"LLM provider: {llm_handler.provider}")
        print(f"Model: {llm_handler.model}")
    else:
        print("❌ Some tests failed. Check the logs for details.")
    print("="*60 + "\n")
    
    return all_success

def test_specific_file(file_path: str, config_file: Optional[str] = None):
    """
    Test metadata extraction on a specific file.
    
    Args:
        file_path: Path to the file to test
        config_file: Optional path to custom config file
    """
    from pathlib import Path
    
    # Check if file exists
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False
    
    # Initialize configuration and LLM handler
    config_manager = get_config_manager(config_file)
    llm_config = config_manager.get_config("llm")
    llm_handler = get_llm_handler(llm_config)
    
    if not llm_handler.is_available():
        logger.error("No LLM provider available. Please check your configuration and dependencies.")
        return False
    
    # Determine file type and select appropriate extractor
    file_ext = Path(file_path).suffix.lower()
    
    try:
        if file_ext in ['.pdf']:
            # Test academic PDF metadata extraction
            if '.pdf' in file_path.lower():
                logger.info("Testing academic PDF metadata extraction")
                
                # Import the PDF metadata extractor
                try:
                    from modules.academic.pdf_metadata import PDFMetadataExtractor
                    extractor = PDFMetadataExtractor(llm_handler=llm_handler)
                except ImportError:
                    logger.warning("Academic module not found, using generic metadata extraction")
                    from core.metadata_extractor import LLMMetadataExtractor
                    extractor = LLMMetadataExtractor(llm_handler=llm_handler)
        
        elif file_ext in ['.epub', '.mobi', '.azw', '.azw3']:
            # Test book metadata extraction
            logger.info("Testing book metadata extraction")
            
            # Import the book metadata extractor
            try:
                from modules.books.book_metadata import BookMetadataExtractor
                extractor = BookMetadataExtractor(llm_handler=llm_handler)
            except ImportError:
                logger.warning("Book module not found, using generic metadata extraction")
                from core.metadata_extractor import LLMMetadataExtractor
                extractor = LLMMetadataExtractor(llm_handler=llm_handler)
        
        else:
            # Use generic metadata extraction
            logger.info("Using generic metadata extraction")
            from core.metadata_extractor import LLMMetadataExtractor
            extractor = LLMMetadataExtractor(llm_handler=llm_handler)
        
        # Extract metadata
        logger.info(f"Extracting metadata from: {file_path}")
        start_time = time.time()
        metadata = extractor.extract_metadata(file_path)
        elapsed_time = time.time() - start_time
        
        # Print results
        print("\n" + "="*60)
        print(f"METADATA EXTRACTION RESULTS FOR: {os.path.basename(file_path)}")
        print("="*60)
        
        if metadata:
            print(f"✅ Extraction succeeded in {elapsed_time:.2f}s")
            print("\nExtracted Metadata:")
            print("-" * 40)
            
            # Print metadata in a neat format
            for key, value in sorted(metadata.items()):
                if key not in ['file_path', 'filename', 'extension']:
                    if isinstance(value, str) and len(value) > 100:
                        print(f"{key}: {value[:100]}...")
                    else:
                        print(f"{key}: {value}")
            
            print("-" * 40)
            
            # If it's a book or PDF, try generating a filename
            if hasattr(extractor, 'generate_filename'):
                try:
                    filename = extractor.generate_filename(metadata)
                    print(f"\nGenerated filename: {filename}")
                except Exception as e:
                    print(f"\nError generating filename: {e}")
        else:
            print(f"❌ Extraction failed or returned empty metadata in {elapsed_time:.2f}s")
        
        print("="*60 + "\n")
        
        return bool(metadata)
        
    except Exception as e:
        logger.error(f"Error testing file: {e}", exc_info=True)
        print(f"❌ Error: {e}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test LLM handler functionality")
    parser.add_argument("--config", help="Path to custom config file")
    parser.add_argument("--provider", choices=["auto", "transformers", "ollama", "none"], 
                        help="LLM provider to use (overrides config)")
    parser.add_argument("--file", help="Specific file to test metadata extraction on")
    
    args = parser.parse_args()
    
    if args.file:
        return 0 if test_specific_file(args.file, args.config) else 1
    else:
        return 0 if test_llm_handler(args.config, args.provider) else 1

if __name__ == "__main__":
    sys.exit(main())
