# Plan: Enhancing Academic Module for Medical Journal Articles and Zotero Integration

**Date:** 2025-05-18

**Objective:** To enhance the SmartFileManager's capabilities to accurately process medical journal articles (PDFs), extract relevant metadata, and integrate seamlessly with a local Zotero library for duplicate detection and management.

**Core Strategy:** We will adopt a hybrid approach, leveraging the existing Zotero integration functionalities of the `academic` module and enhancing the metadata extraction capabilities by adapting the `MedicalMetadataExtractor` from the `medical` module.

## I. Enhancements to `MedicalMetadataExtractor` (in `modules/medical/medical_metadata.py`)

1.  **Add New Document Type:**
    *   Modify `MedicalMetadataExtractor.__init__` to include `"Medical Journal Article"` in `self.document_types`.

2.  **Improve Document Type Detection:**
    *   Modify `MedicalMetadataExtractor._determine_document_type`:
        *   Add logic to identify "Medical Journal Article". This will involve checking for keywords such as "doi:", "pmid:", "abstract", "keywords", "journal of", "references", common journal title patterns, or structural cues if possible.
        *   This check should ideally have higher precedence for PDFs than more generic types if strong indicators are present.

3.  **Create Specialized LLM Extraction for Journal Articles:**
    *   Create a new method: `_extract_journal_article_metadata_llm(self, content: str) -> Dict[str, Any]`.
    *   **LLM Prompt:** This method will use a new, specific LLM prompt designed to extract the following fields from medical journal articles:
        *   Title
        *   Authors (List or comma-separated string)
        *   Journal Name
        *   Publication Year
        *   Volume
        *   Issue
        *   Pages (e.g., "123-130")
        *   DOI (Digital Object Identifier)
        *   PMID (PubMed ID)
        *   Abstract (first few sentences or a summary)
        *   Keywords (List or comma-separated string)
    *   **Response Parsing:** Implement logic to parse the LLM's structured response for these fields.

4.  **Update Main Metadata Extraction Logic:**
    *   Modify `MedicalMetadataExtractor.extract_metadata`:
        *   After `document_type` is determined, if it is `"Medical Journal Article"`, call `_extract_journal_article_metadata_llm(content)` to get metadata.
        *   For other medical document types, continue to use `_extract_medical_metadata_llm(content)` (which handles clinical data).
        *   Ensure appropriate merging/prioritization of metadata from different sources (e.g., basic PDF info, rule-based, LLM-based).

5.  **Normalize Journal Article Metadata:**
    *   Modify `MedicalMetadataExtractor._normalize_medical_metadata`:
        *   Add normalization rules for the new journal article fields (e.g., cleaning author lists, standardizing DOI/PMID formats, year extraction).

6.  **Standardize Filename Generation for Journal Articles:**
    *   Modify `MedicalMetadataExtractor.generate_filename`:
        *   If `metadata["document_type"] == "Medical Journal Article"`, generate a filename primarily based on `Author_Year_Title.pdf` (or a configurable Zotero-like pattern).
        *   This ensures consistency for academic papers.

## II. Integration into `ZoteroIntegration` (in `modules/academic/zotero_integration.py`)

1.  **Switch to Enhanced Metadata Extractor:**
    *   Modify `ZoteroIntegration.__init__`:
        *   Change `self.metadata_extractor = PDFMetadataExtractor(llm_handler=llm_handler)` to `self.metadata_extractor = MedicalMetadataExtractor(llm_handler=llm_handler)`.
        *   Ensure the `llm_handler` is correctly passed and utilized.

2.  **Update Duplicate Detection Logic (If Necessary):**
    *   Review `ZoteroIntegration.find_duplicates`.
    *   The current logic uses file hash and filename similarity. Richer metadata (like DOI or PMID if available from both local extraction and Zotero) could be used for more robust matching in the future, but for now, ensuring the basic metadata (title, author, year for filename generation) is accurate will be the primary focus.

3.  **Update Reporting (If Necessary):**
    *   Review `ZoteroIntegration.generate_duplicate_report`.
    *   Ensure the report can accommodate and display any new, relevant metadata fields if we decide to include them in the comparison details.

## III. Testing and Validation Strategy

1.  **Prepare Test Data:**
    *   Gather a small, diverse set of sample medical journal PDF articles.
    *   Include articles with and without easily identifiable DOIs/PMIDs.

2.  **Unit Testing for `MedicalMetadataExtractor`:**
    *   Test `_determine_document_type` for correct identification of journal articles.
    *   Test `_extract_journal_article_metadata_llm` for accurate extraction of all specified fields (Title, Authors, Journal, Year, DOI, PMID, etc.).
    *   Test `generate_filename` for correct Zotero-style naming for journal articles.

3.  **Integration Testing for `ZoteroIntegration`:**
    *   Test the end-to-end process: scanning a directory of test PDFs, extracting metadata using the enhanced extractor, and comparing against a small, controlled Zotero test library (if feasible to set up, or mock Zotero responses).
    *   Verify duplicate detection accuracy.
    *   Validate the content of the generated duplicate reports.

## IV. Future Considerations (Post-MVP)

*   Allowing user configuration for filename patterns for journal articles.
*   Using DOI/PMID for more definitive duplicate checking against Zotero.
*   Fetching metadata from online databases (e.g., PubMed, CrossRef) using DOI/PMID if local extraction is insufficient.
*   Allowing the user to review and confirm/correct extracted metadata before renaming or processing.

---
