import sqlite3
import os
import shutil
import re
import glob

# --- Configuration ---
ZOTERO_DB_PATH = r"C:\Users\tusharshah\Zotero\zotero.sqlite"
ZOTERO_STORAGE_PATH = r"C:\Users\tusharshah\Zotero\storage"
TARGET_DIR = r"\\mac\tusharshah\ATeam Dropbox\Tushar Shah\Documents\Computer\VS Code\Python Programs\SmartFileManager\modules\academic\academic_test"
MARKDOWN_OUTPUT_PATH = os.path.join(TARGET_DIR, "article_metadata.md")
MAX_ARTICLES = 10
EXCLUDE_ITEM_IDS = [901, 887]

# --- Helper Functions ---
def get_db_connection(db_path):
    """Establishes a read-only connection to the SQLite database."""
    try:
        # Connect in read-only mode
        conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
        conn.row_factory = sqlite3.Row # Access columns by name
        print(f"Successfully connected to Zotero database: {db_path}")
        return conn
    except sqlite3.Error as e:
        print(f"Error connecting to database {db_path}: {e}")
        return None

def get_field_value(conn, item_id, field_name):
    """Retrieves a specific field value for an item using Zotero's EAV model."""
    query = """
        SELECT iv.value
        FROM itemData idata
        JOIN fields f ON idata.fieldID = f.fieldID
        JOIN itemDataValues iv ON idata.valueID = iv.valueID
        WHERE idata.itemID = ? AND f.fieldName = ?
    """
    cursor = conn.cursor()
    cursor.execute(query, (item_id, field_name))
    row = cursor.fetchone()
    return row['value'] if row else None

def get_first_author_last_name(conn, item_id):
    """Retrieves the last name of the first author for an item."""
    query = """
        SELECT c.lastName
        FROM itemCreators ic
        JOIN creators c ON ic.creatorID = c.creatorID
        JOIN creatorTypes ct ON ic.creatorTypeID = ct.creatorTypeID
        WHERE ic.itemID = ? AND ct.creatorType = 'author'
        ORDER BY ic.orderIndex
        LIMIT 1
    """
    cursor = conn.cursor()
    cursor.execute(query, (item_id,))
    row = cursor.fetchone()
    return row['lastName'] if row else None

def get_pdf_attachment_info(conn, parent_item_id):
    """Retrieves the path and original filename for a PDF attachment of an item."""
    query = """
        SELECT i.key, ia.path
        FROM itemAttachments ia
        JOIN items i ON ia.itemID = i.itemID
        WHERE ia.parentItemID = ? 
          AND (ia.contentType = 'application/pdf' OR lower(ia.path) LIKE '%.pdf')
          AND ia.linkMode IN (0, 1) -- LINK_MODE_IMPORTED_FILE, LINK_MODE_IMPORTED_URL
        ORDER BY ia.itemID DESC -- Get the latest attachment if multiple
        LIMIT 1
    """
    # Note: linkMode IN (0,1) usually means Zotero manages the file in its storage.
    # path for linkMode 0,1 is typically 'storage:filename.pdf' or just 'filename.pdf'
    cursor = conn.cursor()
    cursor.execute(query, (parent_item_id,))
    row = cursor.fetchone()
    if row and row['path']:
        attachment_key = row['key']
        original_filename = row['path'].replace('storage:', '')
        # Path in Zotero storage is ZOTERO_STORAGE_PATH / ATTACHMENT_ITEM_KEY / original_filename
        full_path = os.path.join(ZOTERO_STORAGE_PATH, attachment_key, original_filename)
        return full_path, original_filename
    return None, None

def parse_year(date_str):
    """Extracts year from a date string."""
    if not date_str:
        return None
    match = re.search(r'\b(\d{4})\b', date_str)
    return match.group(1) if match else date_str 

def parse_first_page(pages_str):
    """Extracts the first page from a page range string."""
    if not pages_str:
        return None
    match = re.match(r'(\d+)', pages_str)
    return match.group(1) if match else pages_str

# --- Main Logic ---
def main():
    print("Starting Zotero article processing...")

    # Clean up target directory before processing
    print(f"Cleaning up target directory: {TARGET_DIR}")
    if os.path.exists(TARGET_DIR):
        # Delete existing Article*.pdf files
        for pdf_file in glob.glob(os.path.join(TARGET_DIR, "Article *.pdf")):
            try:
                os.remove(pdf_file)
                print(f"  - Deleted existing PDF: {pdf_file}")
            except OSError as e:
                print(f"  - Error deleting file {pdf_file}: {e}")
        # Delete existing markdown file
        if os.path.exists(MARKDOWN_OUTPUT_PATH):
            try:
                os.remove(MARKDOWN_OUTPUT_PATH)
                print(f"  - Deleted existing Markdown file: {MARKDOWN_OUTPUT_PATH}")
            except OSError as e:
                print(f"  - Error deleting Markdown file {MARKDOWN_OUTPUT_PATH}: {e}")
    else:
        print(f"Target directory {TARGET_DIR} does not exist. It will be created.")

    conn = get_db_connection(ZOTERO_DB_PATH)
    if not conn:
        return

    if not os.path.exists(TARGET_DIR):
        try:
            os.makedirs(TARGET_DIR)
            print(f"Created target directory: {TARGET_DIR}")
        except OSError as e:
            print(f"Error creating target directory {TARGET_DIR}: {e}")
            conn.close()
            return

    cursor = conn.cursor()

    # Get itemTypeID for 'journalArticle'
    cursor.execute("SELECT itemTypeID FROM itemTypes WHERE typeName = 'journalArticle'")
    journal_article_type_id_row = cursor.fetchone()
    if not journal_article_type_id_row:
        print("Error: Could not find itemTypeID for 'journalArticle'.")
        conn.close()
        return
    journal_article_type_id = journal_article_type_id_row['itemTypeID']

    # Query for journal articles
    query_articles = """
        SELECT itemID 
        FROM items 
        WHERE itemTypeID = ? AND itemID NOT IN (SELECT itemID FROM deletedItems)
        ORDER BY dateModified DESC -- Or dateAdded, or randomly for variety
        LIMIT ?
    """
    
    articles_data = []
    article_count = 0

    print(f"Fetching up to {MAX_ARTICLES} journal articles, excluding problematic ones...")
    # Fetch more candidates to account for exclusions and missing PDFs
    for row in cursor.execute(query_articles, (journal_article_type_id, MAX_ARTICLES * 3)):
        if article_count >= MAX_ARTICLES:
            break
        
        item_id = row['itemID']

        if item_id in EXCLUDE_ITEM_IDS:
            print(f"\nSkipping excluded itemID: {item_id}")
            continue # Skip this item

        print(f"\nProcessing itemID: {item_id}...")

        title = get_field_value(conn, item_id, 'title')
        first_author_last_name = get_first_author_last_name(conn, item_id)
        journal_title = get_field_value(conn, item_id, 'publicationTitle')
        publication_date = get_field_value(conn, item_id, 'date')
        volume = get_field_value(conn, item_id, 'volume')
        issue = get_field_value(conn, item_id, 'issue')
        pages = get_field_value(conn, item_id, 'pages')

        publication_year = parse_year(publication_date)
        first_page = parse_first_page(pages)

        pdf_path, original_pdf_filename = get_pdf_attachment_info(conn, item_id)

        if not pdf_path or not os.path.exists(pdf_path):
            print(f"  - PDF attachment not found or path invalid for itemID {item_id}. Skipping.")
            continue
        
        print(f"  - Title: {title}")
        print(f"  - PDF found: {pdf_path}")

        article_count += 1
        new_filename = f"Article {article_count}.pdf"
        destination_path = os.path.join(TARGET_DIR, new_filename)

        try:
            shutil.copy2(pdf_path, destination_path)
            print(f"  - Copied to: {destination_path}")
        except Exception as e:
            print(f"  - Error copying file {pdf_path} to {destination_path}: {e}")
            article_count -= 1 # Decrement as copy failed
            continue

        articles_data.append({
            'Original Filename': original_pdf_filename,
            'Article Title': title or '',
            'First Author Last Name': first_author_last_name or '',
            'Journal Title': journal_title or '',
            'Publication Year': publication_year or '',
            'Journal Volume': volume or '',
            'Issue Number': issue or '',
            'First Page': first_page or '',
            'New Filename': new_filename
        })

    conn.close()
    print(f"\nFinished processing items. Found and copied {article_count} articles.")

    if not articles_data:
        print("No articles were processed or copied. Markdown table will not be generated.")
        return

    # Generate Markdown table
    print(f"Generating Markdown table at: {MARKDOWN_OUTPUT_PATH}")
    md_content = "| Original Filename | Article Title | First Author Last Name | Journal Title | Publication Year | Journal Volume | Issue Number | First Page | New Filename |\n"
    md_content += "|-------------------|---------------|------------------------|---------------|------------------|----------------|--------------|------------|--------------|\n"
    for data in articles_data:
        md_content += f"| {data['Original Filename']} | {data['Article Title']} | {data['First Author Last Name']} | {data['Journal Title']} | {data['Publication Year']} | {data['Journal Volume']} | {data['Issue Number']} | {data['First Page']} | {data['New Filename']} |\n"

    try:
        with open(MARKDOWN_OUTPUT_PATH, 'w', encoding='utf-8') as f:
            f.write(md_content)
        print("Markdown table successfully generated.")
    except IOError as e:
        print(f"Error writing Markdown file: {e}")

if __name__ == "__main__":
    main()
