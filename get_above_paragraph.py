import re
from bs4 import BeautifulSoup
import csv
import os
import argparse
from tqdm import tqdm  # Progress bar library
from util import iterate_search_files

# Function to determine if a paragraph represents a date
def is_date(paragraph):
    # Define a pattern for dates (supporting both full-width and half-width numbers)
    date_pattern = r'[０-９0-9]{4}年[０-９0-9]{1,2}月[０-９0-9]{1,2}日'
    return re.search(date_pattern, paragraph) is not None

# Function to clean paragraphs
def clean_paragraph(paragraph):
    # Remove leading numbers (e.g., (1), 4) and spaces (supporting both full-width and half-width)
    paragraph = re.sub(r'^[\(（]?[０-９0-9]+[\)）]?[ 　]*', '', paragraph)
    # Remove special brackets like 【沿革】
    paragraph = re.sub(r'[【】]', '', paragraph)
    return paragraph.strip()

# Function to save data to a CSV file
import os
import csv

def save_to_csv(html_file_path, table_id, elements, csv_file_path):
    # Extract the docid from the table_id (first 8 characters)
    docid = table_id[:8]

    # Set the output CSV file path to include the docid directory
    output_dir = os.path.join(csv_file_path, docid)
    os.makedirs(output_dir, exist_ok=True)
    csv_file_path = os.path.join(output_dir, f"{table_id}_sentences.csv")
    
    # Prepare data to be saved
    row = [elements, f"{table_id}-aboveparagraph", '']

    # Check if the CSV file already exists, and read existing data
    if os.path.exists(csv_file_path):
        with open(csv_file_path, 'r', encoding='utf-8') as f:
            existing_data = list(csv.reader(f))
    else:
        existing_data = [['Sentence', 'ID', 'data']]  # Add header if file doesn't exist
    
    # Insert the new data below the header
    existing_data.insert(1, row)
    
    # Write data back to the CSV file
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(existing_data)

# Function to find the paragraph above a given table
def find_paragraph_above(table):
    # Retrieve all elements above the table (until another table is reached)
    for element in table.find_all_previous():
        # Stop if another table is encountered
        if element.name == 'table':
            break

        # Skip if the current element is part of the <table>
        if table in element.find_all('table'):
            continue

        # Collect text from elements containing text
        text = element.get_text().strip()

        # Verify that the text is not inside another <table>
        if text and not element.find_parent('table'):
            # Return if the extracted paragraph is not a date
            if not is_date(text):
                return text

    return None  # Return None if no valid paragraph is found

# Function to extract elements above tables and save them
def extract_top_elements(html_file_path, csv_file_path):
    # Open and parse the HTML file
    with open(html_file_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')

    # Find all <table> tags in the HTML
    tables = soup.find_all('table')

    # Process each table to find elements above it
    for table in tables:
        table_id = table.get('table-id')
        if not table_id:
            continue  # Skip if there is no table-id

        # Recursively search for paragraphs above the table
        current_table = table
        paragraph = None

        # Continue searching upwards until a valid paragraph is found
        while current_table:
            paragraph = find_paragraph_above(current_table)
            if paragraph:
                break  # Exit the loop once a valid paragraph is found

            # Move to the previous <table> and continue the search
            current_table = current_table.find_previous('table')

        # Display the table-id and the obtained paragraph
        if paragraph and not paragraph.startswith("div#pageDIV"):
            paragraph = "以下の文書は「" + clean_paragraph(paragraph) + "」について記述されています。"
            save_to_csv(html_file_path, table_id, paragraph, csv_file_path)

# Example usage
def main():
    # Command-line argument processing
    parser = argparse.ArgumentParser(description="Process HTML files to extract elements above tables.")
    parser.add_argument('--html', type=str, help="The root directory containing HTML files to process.")
    parser.add_argument('--csv', type=str, help="Directory to store the extracted paragraphs in CSV format.")
    args = parser.parse_args()

    # Execute the process for the root directory of HTML files
    for html_file_path in iterate_search_files(args.html, '.html'):
        extract_top_elements(html_file_path, args.csv)

if __name__ == "__main__":
    main()