# Book Test Reference Document

This document serves as a reference for the test book files used to validate the SmartFileManager's book module functionality. It contains the original filenames and the expected correctly formatted filenames after processing.

## Test Files

The test files were selected from a curated collection with accurate metadata. For testing purposes, these files have been renamed to generic or incorrect names that the system should be able to process and correctly rename.

## Reference Table

| Original Test Filename | Expected Correct Filename | Author | Title | Year | ISBN | Publisher |
|------------------------|--------------------------|--------|-------|------|------|-----------|
| book1.epub | Ishiguro_2005_Never Let Me Go.epub | Kazuo Ishiguro | Never Let Me Go | 2005 | 9781400078776 | Vintage International |
| random-ebook-file.epub | Clarke_1968_2001 A Space Odyssey.epub | Arthur C. Clarke | 2001: A Space Odyssey | 1968 | 9780451457998 | Roc |
| download(3).epub | Orwell_1949_1984.epub | George Orwell | 1984 | 1949 | 9780451524935 | Signet Classics |
| untitled.epub | Dumas_1844_The Count of Monte Cristo.epub | Alexandre Dumas | The Count of Monte Cristo | 1844 | 9780140449266 | Penguin Classics |
| ebook_fiction_2023.epub | Gaiman_2001_American Gods.epub | Neil Gaiman | American Gods | 2001 | 9780380789030 | William Morrow |
| unknown-title.epub | Tolkien_1954_The Fellowship of the Ring.epub | J.R.R. Tolkien | The Fellowship of the Ring | 1954 | 9780618346257 | Houghton Mifflin |
| sci-fi-collection.epub | Herbert_1965_Dune.epub | Frank Herbert | Dune | 1965 | 9780441172719 | Ace Books |
| purchased_item.epub | Austen_1813_Pride and Prejudice.epub | Jane Austen | Pride and Prejudice | 1813 | 9780141439518 | Penguin Classics |
| item5234.epub | Rowling_1997_Harry Potter and the Philosophers Stone.epub | J.K. Rowling | Harry Potter and the Philosopher's Stone | 1997 | 9780747532699 | Bloomsbury |
| literature-classics.epub | Fitzgerald_1925_The Great Gatsby.epub | F. Scott Fitzgerald | The Great Gatsby | 1925 | 9780743273565 | Scribner |

## Test Procedure

1. Copy the original test files to the `test_data/books_test` directory.
2. Run the book module tests on these files.
3. Compare the resulting filenames and metadata against this reference document.
4. Analyze log files for any errors or warnings.

## Expected Results

- Each file should be correctly renamed according to the pattern: `{Author}_{Year}_{Title}.epub`
- Metadata extraction should identify all key fields (Author, Title, Year, ISBN, Publisher)
- No errors should be encountered during processing

## Log Analysis Guide

The test log should be checked for:
1. Successful metadata extraction for each file
2. Accuracy of extracted metadata compared to reference values
3. Proper filename generation based on the template
4. Successful file renaming operations
5. Any warnings or errors that might indicate issues with the extraction process
