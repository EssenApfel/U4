import os
import pandas as pd
from tqdm import tqdm  # プログレスバーのためのライブラリ
#0918.texまでのプログラム
def normalize_text(text):
    import re
    import unicodedata
    normalized_text = unicodedata.normalize('NFKC', text)
    normalized_text = re.sub(r"\s", "", normalized_text)
    normalized_text = normalized_text.replace(',', '')
    triangles = ['▲', '△', '▴', '▵']
    for triangle in triangles:
        normalized_text = normalized_text.replace(triangle, '-')
    if normalized_text == '0百万円':
        normalized_text = '0'
    elif normalized_text.endswith('百万円'):
        normalized_text = normalized_text.replace('百万円', '000000')
    elif normalized_text.endswith('千円'):
        normalized_text = normalized_text.replace('千円', '000')
    normalized_text = normalized_text.rstrip('円株個')
    return normalized_text


def extract_tables_from_html(file_path):
    import pandas as pd
    from bs4 import BeautifulSoup

    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')

    tables = soup.find_all('table', {'table-id': True})
    extracted_tables = {}

    for table in tables:
        table_id = table.get('table-id')
        rows = table.find_all('tr')

        max_columns = 0
        for row in rows:
            current_columns = 0
            cells = row.find_all(['td', 'th'])
            for cell in cells:
                colspan = int(cell.get('colspan', 1))
                current_columns += colspan
            max_columns = max(max_columns, current_columns)

        df_data = [[''] * max_columns for _ in range(len(rows))]

        row_index = 0
        for row in rows:
            col_index = 0
            cells = row.find_all(['td', 'th'])
            for cell in cells:
                if col_index >= max_columns:
                    break

                while col_index < max_columns and df_data[row_index][col_index] != '':
                    col_index += 1

                colspan = int(cell.get('colspan', 1))
                rowspan = int(cell.get('rowspan', 1))
                cell_id = cell.get('cell-id')
                text = cell.get_text(strip=True)
                text = normalize_text(text)
                cell_content = {'text': text, 'id': cell_id}

                for i in range(rowspan):
                    for j in range(colspan):
                        if row_index + i < len(df_data) and col_index + j < max_columns:
                            df_data[row_index + i][col_index + j] = cell_content

                col_index += colspan

            row_index += 1

        df = pd.DataFrame(df_data)
        extracted_tables[table_id] = df

    return extracted_tables


def normalize_dataframes(csv_file, tables):
    import pandas as pd
    
    csv_data = pd.read_csv(csv_file, header=None, dtype=str)
    csv_dict = {row[0]: row[1] for _, row in csv_data.iterrows() if pd.notna(row[1]) and row[1] != ''}

    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    for table_id, df in tables.items():
        processed_cells = set()
        
        for row in range(df.shape[0]):
            for col in range(df.shape[1]):
                cell_value = df.iat[row, col]
                
                if isinstance(cell_value, dict) and 'text' in cell_value:
                    text_value = cell_value['text']

                    for key, csv_value in csv_dict.items():
                        if text_value.endswith(key) and csv_value:
                            for right_col in range(col + 1, df.shape[1]):
                                right_cell_value = df.iat[row, right_col]
                                if isinstance(right_cell_value, dict) and 'text' in right_cell_value:
                                    right_text_value = right_cell_value['text']
                                    if is_number(right_text_value) and (row, right_col) not in processed_cells and right_text_value != "0":
                                        df.iat[row, right_col]['text'] = str(right_text_value) + csv_value
                                        processed_cells.add((row, right_col))
                            for down_row in range(row + 1, df.shape[0]):
                                down_cell_value = df.iat[down_row, col]
                                if isinstance(down_cell_value, dict) and 'text' in down_cell_value:
                                    down_text_value = down_cell_value['text']
                                    if is_number(down_text_value) and (down_row, col) not in processed_cells and down_text_value != "0":
                                        df.iat[down_row, col]['text'] = str(down_text_value) + csv_value
                                        processed_cells.add((down_row, col))
                            processed_cells.add((row, col))

        for row in range(df.shape[0]):
            for col in range(df.shape[1]):
                cell_value = df.iat[row, col]
                
                if isinstance(cell_value, dict) and 'text' in cell_value:
                    text_value = cell_value['text']

                    for key, csv_value in csv_dict.items():
                        if text_value.startswith(key) and csv_value:
                            number_part = text_value[len(key):]
                            if is_number(number_part):
                                df.iat[row, col]['text'] = number_part + csv_value

    return tables


def process_and_save_tables(root_dir, csv_file):
    subdirs = [subdir for subdir in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, subdir))]

    with tqdm(total=len(subdirs), desc="Processing directories") as pbar:
        for subdir in subdirs:
            subdir_path = os.path.join(root_dir, subdir)
            pbar.set_description(f"Processing {subdir}")
            
            html_files = [file for file in os.listdir(subdir_path) if file.endswith('.html')]

            for file in tqdm(html_files, desc="Processing HTML files", leave=False):
                file_path = os.path.join(subdir_path, file)
                tables = extract_tables_from_html(file_path)
                tables = normalize_dataframes(csv_file, tables)

                for table_id, df in tables.items():
                    output_file = os.path.join(subdir_path, f"{table_id}.pkl")
                    df.to_pickle(output_file)
                    # print(f"Saved: {output_file}")
            
            pbar.update(1)


if __name__ == '__main__':
    csv_file = 'old/unit_list.csv'
    root_dir = "/home/takasago/U4/table_retrieval/valid_tr/reports_tr_valid"
    process_and_save_tables(root_dir, csv_file)