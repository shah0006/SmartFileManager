"""
Medical Metadata Extraction Module.

This module extends the core metadata extraction framework
to handle medical files specifically, such as medical reports,
scans, lab results, and medical literature.
"""

import os
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
import datetime
from pathlib import Path

from core.metadata_extractor import LLMMetadataExtractor

# Setup logging
logger = logging.getLogger(__name__)

class MedicalMetadataExtractor(LLMMetadataExtractor):
    """Metadata extractor specialized for medical files."""
    
    def __init__(self, llm_handler=None):
        """
        Initialize the medical metadata extractor.
        
        Args:
            llm_handler: Handler for local LLM interactions
        """
        super().__init__(llm_handler=llm_handler)
        
        # Medical document types
        self.document_types = [
            "Lab Result", "Radiology Report", "Clinical Note", 
            "Discharge Summary", "Prescription", "Medical Image",
            "Medical Literature", "Patient Record", "Medical Invoice",
            "Medical Journal Article"
        ]
        
        # Common medical document headers
        self.header_patterns = {
            "patient_name": [
                r"(?:Patient(?:'s)?\s*(?:Name|ID):?\s*)([\w\s\-\.]+)",
                r"(?:Name:?\s*)([\w\s\-\.]+)",
                r"(?:PATIENT:?\s*)([\w\s\-\.]+)"
            ],
            "patient_id": [
                r"(?:(?:Patient|Medical|Hospital)\s*(?:ID|Number|#):?\s*)([A-Z0-9\-]+)",
                r"(?:MRN:?\s*)([A-Z0-9\-]+)",
                r"(?:ID:?\s*)([A-Z0-9\-]+)"
            ],
            "date": [
                r"(?:Date(?:\s+of)?(?:\s+(?:Service|Report|Visit|Exam|Examination|Admission|Study)):?\s*)(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4})",
                r"(?:DOS:?\s*)(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})",
                r"(?:Report\s+Date:?\s*)(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4})"
            ],
            "physician": [
                r"(?:(?:Physician|Doctor|Provider|Attending|Radiologist|Pathologist):?\s*)((?:Dr\.?|MD|DO)\s*[\w\s\-\.]+)",
                r"(?:(?:Ordered|Referring)\s+(?:by|physician):?\s*)((?:Dr\.?|MD|DO)\s*[\w\s\-\.]+)",
                r"(?:(?:Interpreting|Reading)\s+(?:physician|radiologist):?\s*)((?:Dr\.?|MD|DO)\s*[\w\s\-\.]+)"
            ],
            "facility": [
                r"(?:(?:Hospital|Facility|Clinic|Center|Laboratory|Lab):?\s*)([\w\s\-\.&]+)",
                r"(?:(?:Performed|Location)\s+at:?\s*)([\w\s\-\.&]+)"
            ],
            "exam_type": [
                r"(?:(?:Exam|Examination|Procedure|Study|Test):?\s*)([\w\s\-\.\(\)]+)",
                r"(?:(?:Diagnostic|Laboratory|Lab)\s+(?:Test|Procedure):?\s*)([\w\s\-\.\(\)]+)"
            ]
        }
    
    def get_file_content(self, file_path: str) -> Optional[str]:
        """
        Extract text content from a medical file.
        
        Args:
            file_path: Path to the medical file
            
        Returns:
            Extracted text content or None if extraction fails
        """
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == ".pdf":
                return self._extract_pdf_content(file_path)
            elif file_ext == ".txt":
                return self._extract_text_content(file_path)
            elif file_ext in [".doc", ".docx"]:
                return self._extract_msword_content(file_path)
            elif file_ext in [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".gif"]:
                return self._extract_image_content(file_path)
            elif file_ext == ".dicom" or file_ext == ".dcm":
                return self._extract_dicom_content(file_path)
            elif file_ext in [".html", ".htm", ".xml"]:
                return self._extract_html_content(file_path)
            else:
                logger.warning(f"Unsupported file format for content extraction: {file_ext}")
                return None
        except Exception as e:
            logger.error(f"Error extracting content from {file_path}: {e}")
            return None
    
    def _extract_pdf_content(self, file_path: str) -> str:
        """Extract text content from PDF file."""
        try:
            # First try PyMuPDF (fitz) as it's faster
            try:
                import fitz  # PyMuPDF
                
                text_content = ""
                doc = fitz.open(file_path)
                
                # Extract text from all pages
                for page_num in range(doc.page_count):
                    page = doc[page_num]
                    text_content += page.get_text()
                
                doc.close()
                return text_content
                
            except ImportError:
                logger.warning("PyMuPDF not available, trying other methods")
                
                # Try pdfminer
                try:
                    from pdfminer.high_level import extract_text
                    return extract_text(file_path)
                    
                except ImportError:
                    logger.warning("pdfminer not available")
                    
                    # Try pdfplumber
                    try:
                        import pdfplumber
                        with pdfplumber.open(file_path) as pdf:
                            return "".join(page.extract_text() or "" for page in pdf.pages)
                    except ImportError:
                        logger.warning("pdfplumber not available")
                        return ""
                    
        except Exception as e:
            logger.error(f"Error extracting PDF content: {e}")
            raise
    
    def _extract_text_content(self, file_path: str) -> str:
        """Extract content from plain text file."""
        try:
            # Try utf-8 first
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                return content
            except UnicodeDecodeError:
                # Fall back to system default encoding
                with open(file_path, "r") as f:
                    content = f.read()
                return content
                
        except Exception as e:
            logger.error(f"Error extracting text content: {e}")
            raise
    
    def _extract_msword_content(self, file_path: str) -> str:
        """Extract content from MS Word document."""
        try:
            # Try python-docx for docx files
            if file_path.lower().endswith(".docx"):
                try:
                    import docx
                    doc = docx.Document(file_path)
                    return "\n".join([paragraph.text for paragraph in doc.paragraphs])
                except ImportError:
                    logger.warning("python-docx not available for .docx extraction")
            
            # Try textract as fallback for all Word formats
            try:
                import textract
                return textract.process(file_path).decode("utf-8")
            except ImportError:
                logger.warning("textract not available for Word extraction")
                return ""
                
        except Exception as e:
            logger.error(f"Error extracting MS Word content: {e}")
            return ""
    
    def _extract_image_content(self, file_path: str) -> str:
        """Try to extract text from medical images using OCR."""
        try:
            # Try pytesseract for OCR
            try:
                from PIL import Image
                import pytesseract
                
                img = Image.open(file_path)
                text = pytesseract.image_to_string(img)
                return text
                
            except ImportError:
                logger.warning("pytesseract not available for image OCR")
                return f"[Medical image file: {os.path.basename(file_path)}]"
                
        except Exception as e:
            logger.error(f"Error extracting image content: {e}")
            return f"[Medical image file: {os.path.basename(file_path)}]"
    
    def _extract_dicom_content(self, file_path: str) -> str:
        """Extract metadata from DICOM medical image files."""
        try:
            # Try pydicom for DICOM files
            try:
                import pydicom
                
                ds = pydicom.dcmread(file_path)
                
                # Extract relevant DICOM tags
                metadata_text = "DICOM Medical Image\n"
                
                # Patient info
                if hasattr(ds, "PatientName"):
                    metadata_text += f"Patient Name: {ds.PatientName}\n"
                if hasattr(ds, "PatientID"):
                    metadata_text += f"Patient ID: {ds.PatientID}\n"
                if hasattr(ds, "PatientBirthDate"):
                    metadata_text += f"Patient DOB: {ds.PatientBirthDate}\n"
                
                # Study info
                if hasattr(ds, "StudyDescription"):
                    metadata_text += f"Study: {ds.StudyDescription}\n"
                if hasattr(ds, "StudyDate"):
                    metadata_text += f"Date: {ds.StudyDate}\n"
                if hasattr(ds, "Modality"):
                    metadata_text += f"Modality: {ds.Modality}\n"
                
                # Additional metadata
                if hasattr(ds, "InstitutionName"):
                    metadata_text += f"Institution: {ds.InstitutionName}\n"
                if hasattr(ds, "ReferringPhysicianName"):
                    metadata_text += f"Physician: {ds.ReferringPhysicianName}\n"
                
                return metadata_text
                
            except ImportError:
                logger.warning("pydicom not available for DICOM extraction")
                return f"[DICOM medical image file: {os.path.basename(file_path)}]"
                
        except Exception as e:
            logger.error(f"Error extracting DICOM content: {e}")
            return f"[DICOM medical image file: {os.path.basename(file_path)}]"
    
    def _extract_html_content(self, file_path: str) -> str:
        """Extract content from HTML or XML files."""
        try:
            # Try BeautifulSoup
            try:
                from bs4 import BeautifulSoup
                
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                soup = BeautifulSoup(content, "html.parser")
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.extract()
                
                # Get text
                text = soup.get_text()
                
                # Break into lines and remove leading and trailing space
                lines = (line.strip() for line in text.splitlines())
                
                # Break multi-headlines into a line each
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                
                # Remove blank lines
                text = "\n".join(chunk for chunk in chunks if chunk)
                
                return text
                
            except ImportError:
                logger.warning("BeautifulSoup not available for HTML/XML extraction")
                
                # Basic fallback for HTML files
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                # Simple regex to remove HTML tags
                text = re.sub(r'<[^>]+>', ' ', content)
                text = re.sub(r'\s+', ' ', text).strip()
                
                return text
                
        except Exception as e:
            logger.error(f"Error extracting HTML/XML content: {e}")
            return ""
    
    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a medical file.
        
        Args:
            file_path: Path to the medical file
            
        Returns:
            Dictionary of metadata
        """
        # Call the parent class to extract basic metadata first
        metadata = super().extract_metadata(file_path)
        
        # Extract content
        content = self.get_file_content(file_path)
        
        if content:
            # Extract medical-specific metadata from content using regex patterns
            rule_based_metadata = self._extract_medical_metadata_rule_based(content)
            metadata.update(rule_based_metadata)
            
            # Determine document type
            metadata["document_type"] = self._determine_document_type(content, metadata)
            
            # If LLM handler is available, enhance with LLM extraction
            if self.llm_handler:
                llm_metadata = self._extract_medical_metadata_llm(content)
                
                # Only update with LLM values if they aren't already set by rule-based extraction
                for key, value in llm_metadata.items():
                    if key not in metadata or not metadata[key]:
                        metadata[key] = value
        
        # Normalize and clean metadata
        metadata = self._normalize_medical_metadata(metadata)
        
        return metadata
    
    def _extract_medical_metadata_rule_based(self, content: str) -> Dict[str, Any]:
        """
        Extract medical metadata using rule-based patterns.
        
        Args:
            content: Text content from file
            
        Returns:
            Dictionary of extracted metadata
        """
        metadata = {}
        
        # Apply patterns to extract different metadata fields
        for field, patterns in self.header_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    metadata[field] = match.group(1).strip()
                    break
        
        # Check for special medical data types
        # Look for ICD-10 codes
        icd10_codes = re.findall(r'\b[A-Z]\d{2}(?:\.\d{1,2})?\b', content)
        if icd10_codes:
            metadata["icd10_codes"] = icd10_codes
        
        # Look for CPT codes
        cpt_codes = re.findall(r'\b\d{5}\b', content)
        if cpt_codes:
            metadata["cpt_codes"] = cpt_codes
        
        return metadata
    
    def _extract_medical_metadata_llm(self, content: str) -> Dict[str, Any]:
        """
        Extract medical metadata using LLM.
        
        Args:
            content: Text content from file
            
        Returns:
            Dictionary of extracted metadata
        """
        if not self.llm_handler:
            return {}
        
        try:
            # Prepare medical-specific prompt
            max_length = 3000  # Adjust based on your LLM's capabilities
            truncated_content = content[:max_length] if len(content) > max_length else content
            
            prompt = f"""
            Extract the following medical metadata from this document:
            1. Patient Name (use 'Anonymous' if present but should be private)
            2. Document Type (e.g., Lab Result, Radiology Report, Clinical Note)
            3. Date of Service/Report
            4. Physician Name
            5. Medical Facility
            6. Key Finding or Diagnosis (brief, 1-2 sentences)

            Format your response strictly as:
            PATIENT: [patient name or Anonymous]
            DOCUMENT_TYPE: [document type]
            DATE: [date]
            PHYSICIAN: [physician name]
            FACILITY: [facility name]
            FINDING: [key finding or diagnosis]

            Document content:
            {truncated_content}
            """
            
            # Get response
            response = self.llm_handler.get_response(prompt)
            
            # Parse the response
            metadata = {}
            
            patient_match = re.search(r'PATIENT:\s*(.+?)(?:\r?\n|$)', response)
            if patient_match and patient_match.group(1).strip() and patient_match.group(1).strip().lower() != "n/a":
                metadata["patient_name"] = patient_match.group(1).strip()
            
            doc_type_match = re.search(r'DOCUMENT_TYPE:\s*(.+?)(?:\r?\n|$)', response)
            if doc_type_match and doc_type_match.group(1).strip() and doc_type_match.group(1).strip().lower() != "n/a":
                metadata["document_type"] = doc_type_match.group(1).strip()
            
            date_match = re.search(r'DATE:\s*(.+?)(?:\r?\n|$)', response)
            if date_match and date_match.group(1).strip() and date_match.group(1).strip().lower() != "n/a":
                metadata["date"] = date_match.group(1).strip()
            
            physician_match = re.search(r'PHYSICIAN:\s*(.+?)(?:\r?\n|$)', response)
            if physician_match and physician_match.group(1).strip() and physician_match.group(1).strip().lower() != "n/a":
                metadata["physician"] = physician_match.group(1).strip()
            
            facility_match = re.search(r'FACILITY:\s*(.+?)(?:\r?\n|$)', response)
            if facility_match and facility_match.group(1).strip() and facility_match.group(1).strip().lower() != "n/a":
                metadata["facility"] = facility_match.group(1).strip()
            
            finding_match = re.search(r'FINDING:\s*(.+?)(?:\r?\n|$)', response)
            if finding_match and finding_match.group(1).strip() and finding_match.group(1).strip().lower() != "n/a":
                metadata["key_finding"] = finding_match.group(1).strip()
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error extracting medical metadata with LLM: {e}")
            return {}
    
    def _determine_document_type(self, content: str, metadata: Dict[str, Any]) -> str:
        """
        Determine the type of medical document based on content and metadata.
        
        Args:
            content: Text content from file
            metadata: Existing metadata
            
        Returns:
            Document type string
        """
        # Check if we already have a document type from previous extraction
        if "document_type" in metadata and metadata["document_type"]:
            return metadata["document_type"]
        
        # Use rule-based classification
        content_lower = content.lower()
        
        # Check for lab results
        if re.search(r'(?:lab(?:oratory)?|test)\s+results?|chemistry|hematology|reference\s+range', content_lower):
            return "Lab Result"
        
        # Check for radiology reports
        if re.search(r'(?:radiology|imaging|diagnostic)\s+report|(?:x-ray|xray|ct|mri|ultrasound|sonogram|pet)\s+(?:scan|report|examination)', content_lower):
            return "Radiology Report"
        
        # Check for clinical notes
        if re.search(r'(?:clinical|progress|soap|doctor\'?s?|physician\'?s?)\s+(?:note|assessment|documentation)', content_lower):
            return "Clinical Note"
        
        # Check for discharge summary
        if re.search(r'discharge\s+(?:summary|note)|(?:hospital|inpatient)\s+(?:summary|discharge)', content_lower):
            return "Discharge Summary"
        
        # Check for prescriptions
        if re.search(r'(?:prescription|rx)|(?:dispense|take|sig:)|(?:refill)', content_lower):
            return "Prescription"
        
        # Check if it's just an image file
        file_ext = os.path.splitext(self.file_path)[1].lower()
        if file_ext in [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".gif", ".dicom", ".dcm"]:
            return "Medical Image"
        
        # Default to generic medical document
        return "Medical Document"
    
    def _normalize_medical_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize and clean medical metadata.
        
        Args:
            metadata: Raw metadata dictionary
            
        Returns:
            Cleaned metadata dictionary
        """
        # Make a copy to avoid modifying the original
        cleaned = metadata.copy()
        
        # Normalize dates
        if "date" in cleaned and cleaned["date"]:
            date_str = cleaned["date"]
            cleaned["date"] = self._normalize_date(date_str)
        
        # Normalize patient names (for privacy)
        if "patient_name" in cleaned and cleaned["patient_name"]:
            patient = cleaned["patient_name"]
            # Check if it seems like a real name
            if re.search(r'\b[A-Z][a-z]+\b', patient):
                # Replace with generic identifier
                cleaned["patient_name"] = "Anonymous Patient"
        
        # Normalize document type
        if "document_type" in cleaned and cleaned["document_type"]:
            doc_type = cleaned["document_type"]
            # Try to match to standard document types
            best_match = self._find_best_match(doc_type, self.document_types)
            if best_match:
                cleaned["document_type"] = best_match
        
        return cleaned
    
    def _normalize_date(self, date_str: str) -> str:
        """
        Try to normalize date strings to a standard format (YYYY-MM-DD).
        
        Args:
            date_str: Date string to normalize
            
        Returns:
            Normalized date string
        """
        # Common date formats
        formats = [
            "%m/%d/%Y", "%m-%d-%Y", "%m.%d.%Y",
            "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
            "%Y/%m/%d", "%Y-%m-%d", "%Y.%m.%d",
            "%b %d, %Y", "%B %d, %Y",
            "%d %b %Y", "%d %B %Y",
            "%m/%d/%y", "%m-%d-%y", "%m.%d.%y",
            "%d/%m/%y", "%d-%m-%y", "%d.%m.%y"
        ]
        
        # Try each format
        for fmt in formats:
            try:
                # Parse the date
                date_obj = datetime.datetime.strptime(date_str, fmt)
                # Return standardized format
                return date_obj.strftime("%Y-%m-%d")
            except ValueError:
                continue
        
        # If all formats fail, just return the original
        return date_str
    
    def _find_best_match(self, text: str, options: List[str]) -> Optional[str]:
        """
        Find the best matching option for a text value.
        
        Args:
            text: Text to match
            options: List of standard options
            
        Returns:
            Best matching option or None
        """
        text_lower = text.lower()
        
        # Direct match
        for option in options:
            if text_lower == option.lower():
                return option
        
        # Partial match
        best_match = None
        highest_score = 0
        
        for option in options:
            option_lower = option.lower()
            
            # Calculate a simple matching score
            words1 = set(text_lower.split())
            words2 = set(option_lower.split())
            common_words = words1.intersection(words2)
            
            if common_words:
                score = len(common_words) / max(len(words1), len(words2))
                if score > highest_score:
                    highest_score = score
                    best_match = option
        
        # Return best match if score is reasonable
        if highest_score > 0.3:  # Threshold can be adjusted
            return best_match
        
        return None
    
    def generate_filename(self, metadata: Dict[str, Any]) -> str:
        """
        Generate a standardized filename from medical metadata.
        
        Args:
            metadata: Dictionary of metadata
            
        Returns:
            Standardized filename
        """
        # Extract components for filename
        doc_type = metadata.get("document_type", "Medical")
        
        # Use an anonymized patient identifier if available
        patient_id = metadata.get("patient_id", "")
        if patient_id:
            # Mask part of the ID for privacy if it's long enough
            if len(patient_id) > 4:
                patient_id = patient_id[-4:]  # Just use last 4 characters
                patient_id = f"Patient-{patient_id}"
            else:
                patient_id = f"Patient-{patient_id}"
        else:
            patient_id = "Anonymous"
        
        # Format date part
        date_str = metadata.get("date", "")
        if date_str:
            # Try to extract just the date part in a standard format
            date_parts = re.split(r'[/\-\.]', date_str)
            if len(date_parts) == 3:
                # Find which part is likely the year (assume it's the longest)
                if len(date_parts[0]) == 4:  # YYYY-MM-DD
                    date_str = date_parts[0]  # Just use year
                elif len(date_parts[2]) == 4:  # MM-DD-YYYY
                    date_str = date_parts[2]  # Just use year
                else:
                    # If no clear year, use the whole date string
                    date_str = re.sub(r'[/\-\.\s]', '', date_str)
            else:
                # If date format is unclear, just remove separators
                date_str = re.sub(r'[/\-\.\s]', '', date_str)
        
        # Get original extension
        original_ext = metadata.get("extension", ".pdf")
        
        # Clean components for filename use
        doc_type = self._clean_for_filename(doc_type)
        patient_id = self._clean_for_filename(patient_id)
        date_str = self._clean_for_filename(date_str)
        
        # Format: DocType_PatientID_Date.ext
        if date_str:
            filename = f"{doc_type}_{patient_id}_{date_str}{original_ext}"
        else:
            filename = f"{doc_type}_{patient_id}{original_ext}"
        
        return filename
    
    def _clean_for_filename(self, text: str) -> str:
        """
        Clean text for use in a filename.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text suitable for filenames
        """
        # Replace special characters with spaces
        text = re.sub(r'[^\w\s-]', ' ', text)
        
        # Replace spaces with underscores
        text = re.sub(r'\s+', '_', text.strip())
        
        # Remove consecutive underscores
        text = re.sub(r'_+', '_', text)
        
        return text
