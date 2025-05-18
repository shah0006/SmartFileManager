#!/usr/bin/env python3
"""
SmartFileManager Test Runner

This script runs all the test modules for the SmartFileManager project
and generates a comprehensive report on system functionality.
"""

import sys
import os
import argparse
import subprocess
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add parent directory to path so we can import from the project
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test-runner")

# ANSI colors for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def get_test_modules() -> List[Dict[str, Any]]:
    """
    Get a list of all test modules in the tests directory.
    
    Returns:
        List of dictionaries with test module information
    """
    tests_dir = Path(__file__).parent
    modules = []
    
    # Built-in tests
    modules.extend([
        {
            "name": "Dependencies Check",
            "script": "check_dependencies.py",
            "priority": 1,  # Run first
            "required": True,
            "description": "Verifies all required dependencies are available",
            "args": []
        },
        {
            "name": "LLM Handler Test",
            "script": "test_llm_handler.py",
            "priority": 2,
            "required": False,
            "description": "Tests LLM integration for metadata extraction",
            "args": []
        },
        {
            "name": "Book Module Test",
            "script": "test_book_module.py",
            "priority": 3,
            "required": False,
            "description": "Tests the book module functionality",
            "args": []
        }
    ])
    
    # Find other potential test modules
    for file in tests_dir.glob("test_*.py"):
        filename = file.name
        if filename not in [m["script"] for m in modules]:
            modules.append({
                "name": " ".join(filename[5:-3].split("_")).title(),
                "script": filename,
                "priority": 10,  # Lower priority for auto-discovered tests
                "required": False,
                "description": f"Test module for {filename[5:-3].replace('_', ' ')}",
                "args": []
            })
    
    # Sort modules by priority
    modules.sort(key=lambda m: m["priority"])
    
    return modules

def find_test_data_directory() -> Optional[str]:
    """
    Find a directory with test data for running tests.
    
    Returns:
        Path to test data directory or None if not found
    """
    # Places to look for test data
    potential_locations = [
        os.path.join(parent_dir, "test_data"),
        os.path.join(parent_dir, "tests", "test_data"),
        os.path.join(parent_dir, "data", "test"),
        os.path.join(os.path.expanduser("~"), "Documents", "test_books"),
        os.path.join(os.path.expanduser("~"), "Downloads")
    ]
    
    for location in potential_locations:
        if os.path.isdir(location):
            # Check if directory contains potential test files
            book_extensions = ['.epub', '.mobi', '.azw', '.azw3', '.pdf']
            has_test_files = False
            
            for root, _, files in os.walk(location):
                if any(f.lower().endswith(tuple(book_extensions)) for f in files):
                    has_test_files = True
                    break
            
            if has_test_files:
                return location
    
    return None

def run_test_module(module: Dict[str, Any], config_file: Optional[str] = None, 
                   test_data_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Run a single test module and capture its output.
    
    Args:
        module: Test module dictionary
        config_file: Optional path to config file
        test_data_dir: Optional path to test data directory
        
    Returns:
        Dictionary with test results
    """
    logger.info(f"Running test module: {module['name']}")
    
    # Build command
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), module["script"])
    command = [sys.executable, script_path]
    
    # Add arguments
    if config_file:
        command.extend(["--config", config_file])
    
    # Add test data directory for certain tests
    if test_data_dir and module["script"] in ["test_book_module.py"]:
        command.extend(["--dir", test_data_dir])
    
    # Add any module-specific arguments
    command.extend(module["args"])
    
    # Run the test
    start_time = time.time()
    
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=300  # 5-minute timeout
        )
        
        elapsed_time = time.time() - start_time
        
        return {
            "name": module["name"],
            "script": module["script"],
            "success": process.returncode == 0,
            "time_taken": elapsed_time,
            "output": process.stdout,
            "error": process.stderr,
            "returncode": process.returncode
        }
    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        return {
            "name": module["name"],
            "script": module["script"],
            "success": False,
            "time_taken": elapsed_time,
            "output": "",
            "error": "Test timed out after 5 minutes",
            "returncode": -1
        }
    except Exception as e:
        elapsed_time = time.time() - start_time
        return {
            "name": module["name"],
            "script": module["script"],
            "success": False,
            "time_taken": elapsed_time,
            "output": "",
            "error": str(e),
            "returncode": -1
        }

def run_all_tests(args) -> List[Dict[str, Any]]:
    """
    Run all test modules.
    
    Args:
        args: Command-line arguments
        
    Returns:
        List of test result dictionaries
    """
    modules = get_test_modules()
    results = []
    
    # Find test data directory if not specified
    test_data_dir = args.data_dir if args.data_dir else find_test_data_directory()
    if not test_data_dir and not args.skip_data_tests:
        logger.warning("No test data directory found. File-based tests may fail.")
    
    # If specific tests were requested, filter modules
    if args.tests:
        test_names = [t.lower() for t in args.tests]
        modules = [m for m in modules 
                  if m["script"].lower() in test_names or 
                  m["name"].lower() in test_names or
                  m["script"][:-3].lower() in test_names]
        
        if not modules:
            logger.error(f"No test modules found matching: {args.tests}")
            return []
    
    # Run each test module
    for module in modules:
        # Skip file-based tests if requested and no data directory available
        if args.skip_data_tests and module["script"] in ["test_book_module.py"] and not test_data_dir:
            logger.info(f"Skipping {module['name']} (no test data directory)")
            continue
        
        print(f"\n{Colors.BOLD}{Colors.BLUE}Running {module['name']}...{Colors.END}")
        result = run_test_module(module, args.config, test_data_dir)
        results.append(result)
        
        # Print brief result
        if result["success"]:
            print(f"{Colors.GREEN}✓ {module['name']} passed in {result['time_taken']:.2f}s{Colors.END}")
        else:
            print(f"{Colors.RED}✗ {module['name']} failed in {result['time_taken']:.2f}s{Colors.END}")
        
        # Stop on first failure if requested
        if args.fail_fast and not result["success"]:
            print(f"{Colors.YELLOW}Stopping after first failure (--fail-fast){Colors.END}")
            break
    
    return results

def generate_report(results: List[Dict[str, Any]], output_file: Optional[str] = None):
    """
    Generate a test report from the results.
    
    Args:
        results: List of test result dictionaries
        output_file: Optional path to output file
    """
    if not results:
        print(f"{Colors.RED}No test results to report.{Colors.END}")
        return
    
    # Calculate summary statistics
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r["success"])
    failed_tests = total_tests - passed_tests
    total_time = sum(r["time_taken"] for r in results)
    
    # Generate report header
    report = []
    report.append("="*80)
    report.append(f"SMARTFILEMANAGER TEST REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("="*80)
    report.append("")
    report.append(f"Summary: {passed_tests}/{total_tests} tests passed ({(passed_tests/total_tests)*100:.1f}%)")
    report.append(f"Total time: {total_time:.2f} seconds")
    report.append("")
    report.append("-"*80)
    
    # Add individual test results
    for i, result in enumerate(results, 1):
        status = "PASSED" if result["success"] else "FAILED"
        status_color = Colors.GREEN if result["success"] else Colors.RED
        
        report.append(f"\n{i}. {status_color}{status}{Colors.END}: {result['name']}")
        report.append(f"   Script: {result['script']}")
        report.append(f"   Time: {result['time_taken']:.2f} seconds")
        
        if result["error"]:
            report.append(f"   {Colors.RED}Error:{Colors.END} {result['error'][:200]}")
        
        # Include a brief output summary if available
        if result["output"]:
            output_lines = result["output"].splitlines()
            summary_lines = []
            
            # Look for summary sections in output
            in_summary = False
            for line in output_lines:
                if "SUMMARY" in line or "RESULTS" in line or "====" in line:
                    in_summary = True
                    summary_lines.append(line)
                elif in_summary and line.strip():
                    summary_lines.append(line)
                elif in_summary and not line.strip():
                    in_summary = False
            
            # If no summary found, just use the last few lines
            if not summary_lines and len(output_lines) > 5:
                summary_lines = output_lines[-5:]
            
            if summary_lines:
                report.append("\n   Output Summary:")
                for line in summary_lines[:10]:  # Limit to 10 lines
                    report.append(f"     {line}")
                if len(summary_lines) > 10:
                    report.append(f"     ... ({len(summary_lines) - 10} more lines)")
    
    # Generate overall assessment
    report.append("\n" + "="*80)
    if passed_tests == total_tests:
        report.append(f"{Colors.GREEN}{Colors.BOLD}ALL TESTS PASSED{Colors.END}")
        report.append("The SmartFileManager application appears to be fully functional.")
    elif passed_tests >= total_tests * 0.8:
        report.append(f"{Colors.YELLOW}{Colors.BOLD}MOST TESTS PASSED{Colors.END}")
        report.append("The SmartFileManager application is functional but some features may be limited.")
        report.append(f"Failed tests: {', '.join(r['name'] for r in results if not r['success'])}")
    else:
        report.append(f"{Colors.RED}{Colors.BOLD}SIGNIFICANT TEST FAILURES{Colors.END}")
        report.append("The SmartFileManager application may not function correctly.")
        report.append(f"Failed tests: {', '.join(r['name'] for r in results if not r['success'])}")
    report.append("="*80)
    
    # Print report
    report_text = "\n".join(report)
    print(f"\n{report_text}")
    
    # Save report to file if requested
    if output_file:
        try:
            # Strip ANSI color codes for file output
            import re
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            clean_report = ansi_escape.sub('', report_text)
            
            with open(output_file, 'w') as f:
                f.write(clean_report)
            
            print(f"\nReport saved to: {output_file}")
        except Exception as e:
            print(f"{Colors.RED}Error saving report: {e}{Colors.END}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run tests for SmartFileManager")
    parser.add_argument("--config", help="Path to custom config file")
    parser.add_argument("--tests", nargs="+", help="Specific tests to run")
    parser.add_argument("--data-dir", help="Directory with test data files")
    parser.add_argument("--report", help="Save test report to file")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first test failure")
    parser.add_argument("--skip-data-tests", action="store_true", 
                       help="Skip tests requiring data files")
    parser.add_argument("--list", action="store_true", 
                       help="List available test modules without running them")
    
    args = parser.parse_args()
    
    # If --list flag is set, just list the available tests
    if args.list:
        modules = get_test_modules()
        print(f"\n{Colors.BOLD}Available Test Modules:{Colors.END}")
        for i, module in enumerate(modules, 1):
            req_str = f"{Colors.RED}(required){Colors.END}" if module["required"] else ""
            print(f"{i}. {Colors.CYAN}{module['name']}{Colors.END} {req_str}")
            print(f"   Script: {module['script']}")
            print(f"   Description: {module['description']}")
        return 0
    
    try:
        # Run tests
        results = run_all_tests(args)
        
        # Generate report
        if results:
            generate_report(results, args.report)
            
            # Return success if all tests passed
            all_passed = all(r["success"] for r in results)
            return 0 if all_passed else 1
        else:
            print(f"{Colors.RED}No tests were run.{Colors.END}")
            return 1
        
    except Exception as e:
        logger.error(f"Error running tests: {e}", exc_info=True)
        print(f"{Colors.RED}Error running tests: {e}{Colors.END}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
