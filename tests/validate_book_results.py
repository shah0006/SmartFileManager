#!/usr/bin/env python3
"""
Book Test Validation Script

This script compares the results of book module testing against
the reference document to validate metadata extraction and renaming.
"""

import os
import re
import logging
import argparse
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("book-validation")

class BookTestValidator:
    """Validates book module test results against reference data."""
    
    def __init__(self, 
                 test_dir: str, 
                 reference_file: str,
                 results_dir: Optional[str] = None):
        """
        Initialize the validator.
        
        Args:
            test_dir: Directory containing the test files
            reference_file: Path to the reference markdown file
            results_dir: Optional directory containing result files (if different from test_dir)
        """
        self.test_dir = Path(test_dir)
        self.reference_file = Path(reference_file)
        self.results_dir = Path(results_dir) if results_dir else self.test_dir
        
        # Load reference data
        self.reference_data = self._load_reference_data()
        
        # Results of validation
        self.results = {
            "files_checked": 0,
            "files_matched": 0,
            "files_mismatched": 0,
            "details": []
        }
    
    def _load_reference_data(self) -> pd.DataFrame:
        """
        Load the reference data from markdown file.
        
        Returns:
            DataFrame containing reference data
        """
        try:
            # Read the markdown table
            reference_df = pd.read_html(self.reference_file)[0]
            logger.info(f"Loaded reference data with {len(reference_df)} entries")
            return reference_df
        except Exception as e:
            logger.error(f"Error loading reference data: {e}")
            # Try manual parsing as fallback
            return self._parse_markdown_table()
    
    def _parse_markdown_table(self) -> pd.DataFrame:
        """
        Parse the markdown table manually.
        
        Returns:
            DataFrame containing reference data
        """
        with open(self.reference_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract the table section
        table_match = re.search(r'\|.*\|\n\|[-\s|]+\|(.*?)(?=\n\n|\Z)', content, re.DOTALL)
        if not table_match:
            logger.error("Could not find table in reference file")
            return pd.DataFrame()
        
        table_text = table_match.group(0)
        lines = [line.strip() for line in table_text.split('\n')]
        
        # Extract headers
        headers = [h.strip() for h in lines[0].split('|')]
        headers = [h for h in headers if h]
        
        # Extract data rows
        data = []
        for line in lines[2:]:  # Skip header and separator
            if not line or not '|' in line:
                continue
            
            row = [cell.strip() for cell in line.split('|')]
            row = [cell for cell in row if cell != '']
            
            if len(row) == len(headers):
                data.append(row)
        
        return pd.DataFrame(data, columns=headers)
    
    def validate(self) -> Dict[str, Any]:
        """
        Validate test results against reference data.
        
        Returns:
            Dictionary with validation results
        """
        logger.info("Starting validation")
        
        # Check if each expected file exists
        for _, row in self.reference_data.iterrows():
            original_filename = row['Original Test Filename']
            expected_filename = row['Expected Correct Filename']
            
            original_file_path = self.test_dir / original_filename
            result_file_path = self.results_dir / expected_filename
            
            self.results["files_checked"] += 1
            
            # Check if the original file exists
            if not original_file_path.exists() and not result_file_path.exists():
                logger.warning(f"Test file not found: {original_filename}")
                self.results["details"].append({
                    "original": original_filename,
                    "expected": expected_filename,
                    "found": False,
                    "renamed": False,
                    "status": "missing"
                })
                continue
            
            # Check if the file was renamed correctly
            if result_file_path.exists():
                logger.info(f"File correctly renamed: {original_filename} → {expected_filename}")
                self.results["files_matched"] += 1
                self.results["details"].append({
                    "original": original_filename,
                    "expected": expected_filename,
                    "found": True,
                    "renamed": True,
                    "status": "success"
                })
            else:
                # File was not renamed as expected
                self.results["files_mismatched"] += 1
                
                # Check if it was renamed to something else
                possible_renamed_files = self._find_possible_renamed_files(original_filename)
                if possible_renamed_files:
                    actual_filename = possible_renamed_files[0]
                    logger.warning(f"File renamed incorrectly: {original_filename} → {actual_filename} (expected: {expected_filename})")
                    self.results["details"].append({
                        "original": original_filename,
                        "expected": expected_filename,
                        "actual": actual_filename,
                        "found": True,
                        "renamed": True,
                        "status": "incorrect_rename"
                    })
                else:
                    logger.warning(f"File not renamed: {original_filename}")
                    self.results["details"].append({
                        "original": original_filename,
                        "expected": expected_filename,
                        "found": True,
                        "renamed": False,
                        "status": "not_renamed"
                    })
        
        # Calculate success rate
        success_rate = (self.results["files_matched"] / self.results["files_checked"]) * 100 if self.results["files_checked"] > 0 else 0
        self.results["success_rate"] = success_rate
        
        logger.info(f"Validation completed: {self.results['files_matched']}/{self.results['files_checked']} files matched ({success_rate:.1f}%)")
        return self.results
    
    def _find_possible_renamed_files(self, original_filename: str) -> List[str]:
        """
        Find files that might be renamed versions of the original file.
        
        Args:
            original_filename: Original test filename
            
        Returns:
            List of possible renamed files
        """
        # Extract the base name and extension
        base_name = os.path.splitext(original_filename)[0]
        extension = os.path.splitext(original_filename)[1]
        
        # Look for files with similar patterns
        renamed_files = []
        for file in os.listdir(self.results_dir):
            if file != original_filename and file.endswith(extension):
                renamed_files.append(file)
        
        return renamed_files
    
    def generate_report(self, output_file: Optional[str] = None) -> str:
        """
        Generate a validation report.
        
        Args:
            output_file: Optional path to output file
            
        Returns:
            Report content as string
        """
        report_lines = []
        report_lines.append("# Book Module Validation Report")
        report_lines.append(f"Generated at: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Summary
        report_lines.append("## Summary")
        report_lines.append(f"- Files checked: {self.results['files_checked']}")
        report_lines.append(f"- Files matched: {self.results['files_matched']}")
        report_lines.append(f"- Files mismatched: {self.results['files_mismatched']}")
        report_lines.append(f"- Success rate: {self.results.get('success_rate', 0):.1f}%")
        report_lines.append("")
        
        # Details table
        report_lines.append("## Detailed Results")
        report_lines.append("")
        report_lines.append("| Original Filename | Expected Filename | Status | Notes |")
        report_lines.append("|------------------|------------------|--------|-------|")
        
        for detail in self.results["details"]:
            status_icon = "✅" if detail["status"] == "success" else "❌"
            notes = ""
            
            if detail["status"] == "missing":
                notes = "File not found"
            elif detail["status"] == "not_renamed":
                notes = "File not processed"
            elif detail["status"] == "incorrect_rename":
                notes = f"Renamed to: {detail.get('actual', 'unknown')}"
            
            report_lines.append(f"| {detail['original']} | {detail['expected']} | {status_icon} | {notes} |")
        
        # Recommendations
        report_lines.append("")
        report_lines.append("## Recommendations")
        
        if self.results["files_mismatched"] > 0:
            report_lines.append("")
            report_lines.append("Based on the validation results, consider the following:")
            report_lines.append("")
            
            if any(d["status"] == "missing" for d in self.results["details"]):
                report_lines.append("- Ensure all test files are correctly placed in the test directory")
            
            if any(d["status"] == "not_renamed" for d in self.results["details"]):
                report_lines.append("- Check if the book module correctly processes all supported file formats")
                report_lines.append("- Review log files for any errors during metadata extraction")
            
            if any(d["status"] == "incorrect_rename" for d in self.results["details"]):
                report_lines.append("- Examine the metadata extraction logic for accuracy issues")
                report_lines.append("- Verify that the naming template is correctly applied")
        else:
            report_lines.append("")
            report_lines.append("All files were processed correctly. The book module is functioning as expected.")
        
        # Compile the report
        report_content = "\n".join(report_lines)
        
        # Save to file if requested
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                logger.info(f"Report saved to: {output_file}")
            except Exception as e:
                logger.error(f"Error saving report: {e}")
        
        return report_content

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Validate book module test results")
    parser.add_argument("--test-dir", required=True, help="Directory containing test files")
    parser.add_argument("--reference", default=None, help="Path to reference markdown file")
    parser.add_argument("--results-dir", help="Directory containing result files (if different from test dir)")
    parser.add_argument("--report", help="Save validation report to file")
    
    args = parser.parse_args()
    
    # Determine reference file path if not provided
    reference_file = args.reference
    if not reference_file:
        # Look for a file named *reference*.md in the test directory
        test_dir = Path(args.test_dir)
        for file in test_dir.parent.glob("*reference*.md"):
            reference_file = file
            break
        
        if not reference_file:
            reference_file = test_dir.parent / "books_test_reference.md"
    
    try:
        # Initialize validator
        validator = BookTestValidator(
            test_dir=args.test_dir,
            reference_file=reference_file,
            results_dir=args.results_dir
        )
        
        # Run validation
        validator.validate()
        
        # Generate and print report
        report_file = args.report if args.report else os.path.join(args.test_dir, "validation_report.md")
        report = validator.generate_report(report_file)
        print(report)
        
        # Return success if all files matched
        return 0 if validator.results["files_mismatched"] == 0 else 1
        
    except Exception as e:
        logger.error(f"Error during validation: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    exit(main())
