import argparse
import re
import os
import pandas as pd
from tqdm import tqdm
from decimal import Decimal, getcontext, InvalidOperation
import unicodedata
from bs4 import BeautifulSoup
from util import iterate_search_files



def normalize_text(text):
    # 提供されたnormalize関数

    # 全角を半角に揃える
    normalized_text = unicodedata.normalize('NFKC', text)
    # 空白文字を削除する
    normalized_text = re.sub(r"\s", "", normalized_text)
    # カンマを削除する
    normalized_text = normalized_text.replace(',', '')
    # 三角記号をマイナス記号に統一する
    triangles = ['▲', '△', '▴', '▵']
    for triangle in triangles:
        normalized_text = normalized_text.replace(triangle, '-')
    # 位取りの置換
    if normalized_text == '0百万円':
        normalized_text = '0'
    elif normalized_text.endswith('百万円'):
        normalized_text = normalized_text.replace('百万円', '000000')
    elif normalized_text.endswith('千円'):
        normalized_text = normalized_text.replace('千円', '000')
    # 末尾の「円」「株」「個」を削除する
    normalized_text = normalized_text.rstrip('円株個')
    return normalized_text

def normalize_stock_units(text):
    # 「株式の状況(1単元の株式数100株)」を100に変換する関数
    match = re.search(r'株式の状況\(1単元の株式数(\d+)株\)', text)
    if match:
        return match.group(1)
    return text

# 1022, textが消えるのを修正
# def normalize_date(text):
#     # 日付をYYYY-MM-DD形式に正規化する関数
#     match = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', text)
#     if match:
#         year, month, day = match.groups()
#         return f'{year}-{int(month):02}-{int(day):02}'  # 月と日を2桁にする
#     return text

def normalize_date(text):
    import re
    # 日付をYYYY-MM-DD形式に正規化する関数
    match = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', text)
    if match:
        year, month, day = match.groups()
        normalized_date = f'{year}-{int(month):02}-{int(day):02}'  # 月と日を2桁にする
        return text.replace(match.group(0), normalized_date)  # 元のテキストの該当部分を置き換える
    return text

# 上と同様
# def normalize_birth_date(text):
#     # 「生」を無視した日付正規化関数
#     match = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日生', text)
#     if match:
#         year, month, day = match.groups()
#         return f'{year}-{int(month):02}-{int(day):02}'  # 「生」を無視し、日付を2桁にする
#     return text

def normalize_birth_date(text):
    # 「生」を無視した日付正規化関数
    match = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日生', text)
    if match:
        year, month, day = match.groups()
        normalized_date = f'{year}-{int(month):02}-{int(day):02}'  # 月と日を2桁にする
        return text.replace(match.group(0), normalized_date)  # 元のテキストの該当部分を置き換える
    return text

def normalize_currency(text):
    # 「円」と「銭」を小数に正規化する関数
    match = re.search(r'(\d+)円(\d+)銭', text)
    if match:
        yen, sen = match.groups()
        return f'{yen}.{int(sen):02}'
    return text



def extract_tables_from_html(file_path):
    # htmlファイルからtableを抽出するプログラム
    with open(file_path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file, 'html.parser')

    tables = soup.find_all('table', {'table-id': True})
    extracted_tables = {}

    for table in tables:
        table_id = table.get('table-id')
        data = []
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
                # 列のインデックスが範囲内に収まるようにチェック
                if col_index >= max_columns:
                    break

                while col_index < max_columns and df_data[row_index][col_index] != '':  # 空きセルを探す
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




def normalize_dataframes(line_list_csv, unit_list_csv, tables):
    #与えられたtablesをcsvをもとに正規化するプログラム
    
    # 浮動小数点誤差を避けるために、Decimalの精度を設定
    getcontext().prec = 20

    # CSVファイルを辞書形式にロード
    line_list_data = pd.read_csv(line_list_csv, header=None, dtype=str)
    unit_list_data = pd.read_csv(unit_list_csv, header=None, dtype=str)

    # CSVデータを辞書に変換（空の値は無視）
    line_list_dict = {row[0]: Decimal(row[1]) for _, row in line_list_data.iterrows() if pd.notna(row[1]) and row[1] != ''}
    unit_list_dict = {row[0]: Decimal(row[1]) for _, row in unit_list_data.iterrows() if pd.notna(row[1]) and row[1] != ''}

    # 文字列がDecimalに変換できるか確認する関数
    def to_decimal(s):
        try:
            return Decimal(s)
        except (InvalidOperation, TypeError):
            return None

    # "※2"のようなパターンが数字に続いている場合に削除する関数
    def remove_special_marker(text):
        match = re.match(r'※\d(\d+)', text)
        if match:
            return match.group(1)
        return text

    # "〔"と"〕"で囲まれた数字を削除する関数
    def remove_bracketed_numbers(text):
        return re.sub(r'〔\d+〕', '', text)

    # 各テーブルごとのフラグを保持する辞書
    unit_million_flags = []
    flags = []
    table_ids = []

    for table_id, df in tables.items():
        table_ids.append(table_id)
        processed_cells = set()  # 処理済みセルを追跡するセット
        flag = False  # 現在のテーブル用のフラグを初期化
        unit_million_flag = False

        # 各セルのテキストを正規化する（normalize_textをここで適用）
        for row in range(df.shape[0]):
            for col in range(df.shape[1]):
                cell_value = df.iat[row, col]
                if isinstance(cell_value, dict) and 'text' in cell_value:
                    cell_value['text'] = normalize_text(cell_value['text'])

        # CSV処理の前に"※2"や"〔9〕"パターンを削除
        for row in range(df.shape[0]):
            for col in range(df.shape[1]):
                cell_value = df.iat[row, col]
                
                if isinstance(cell_value, dict) and 'text' in cell_value:
                    text_value = cell_value['text']

                    # "※2"パターンを削除
                    text_value = remove_special_marker(text_value)
                    
                    # "〔9〕"のような括弧付き数字を削除
                    text_value = remove_bracketed_numbers(text_value)

                    # 修正後のテキストをセルに反映
                    cell_value['text'] = text_value  

        # line_list.csvに基づく行/列単位の処理
        for row in range(df.shape[0]):
            for col in range(df.shape[1]):
                cell_value = df.iat[row, col]
                
                if isinstance(cell_value, dict) and 'text' in cell_value:
                    text_value = cell_value['text']

                    # line_list.csvルールを適用（テキストがキーで終わるか確認）
                    for key, multiplier in line_list_dict.items():
                        if text_value.endswith(key):
                            # 同じ行の右側のセルを処理
                            for right_col in range(col + 1, df.shape[1]):
                                right_cell_value = df.iat[row, right_col]
                                if isinstance(right_cell_value, dict) and 'text' in right_cell_value:
                                    right_text_value = right_cell_value['text']
                                    right_number = to_decimal(right_text_value)
                                    if right_number is not None and (row, right_col) not in processed_cells:
                                        df.iat[row, right_col]['text'] = str(right_number * multiplier)
                                        processed_cells.add((row, right_col))
                                        flag = True  # 正規化が発生した場合、フラグをTrueにする

                            # 同じ列の下側のセルを処理
                            for down_row in range(row + 1, df.shape[0]):
                                down_cell_value = df.iat[down_row, col]
                                if isinstance(down_cell_value, dict) and 'text' in down_cell_value:
                                    down_text_value = down_cell_value['text']
                                    down_number = to_decimal(down_text_value)
                                    if down_number is not None and (down_row, col) not in processed_cells:
                                        df.iat[down_row, col]['text'] = str(down_number * multiplier)
                                        processed_cells.add((down_row, col))
                                        flag = True  # 正規化が発生した場合、フラグをTrueにする

                            processed_cells.add((row, col))  # 元のセルを処理済みとマーク
                            flag = True  # 正規化が発生した場合、フラグをTrueにする

        # unit_list_top.csvに基づく個別のセル処理
        for row in range(df.shape[0]):
            for col in range(df.shape[1]):
                cell_value = df.iat[row, col]

                if isinstance(cell_value, dict) and 'text' in cell_value:
                    text_value = cell_value['text']

                    # unit_list_top.csvルールを適用（テキストがキーで始まるか確認）
                    for key, multiplier in unit_list_dict.items():
                        if text_value.startswith(key):
                            # キーの後ろの数字部分を抽出
                            number_part = text_value[len(key):]
                            number_value = to_decimal(number_part)
                            if number_value is not None:
                                # 数値を掛け算してテキストを更新
                                df.iat[row, col]['text'] = str(number_value * multiplier)
                                flag = True  # 正規化が発生した場合、フラグをTrueにする
        flags.append(flag)

        # 以前の正規化が行われていない場合、"(単位:百万円)"をチェック
        if not flag:
            for row in range(df.shape[0]):
                for col in range(df.shape[1]):
                    cell_value = df.iat[row, col]
                    
                    if isinstance(cell_value, dict) and 'text' in cell_value:
                        text_value = cell_value['text']
                        
                        if "(単位:百万円)" in text_value:
                            # "(単位:百万円)"が見つかった場合、全ての数値セルに100万倍を適用
                            for r in range(df.shape[0]):
                                for c in range(df.shape[1]):
                                    cell = df.iat[r, c]
                                    if isinstance(cell, dict) and 'text' in cell:
                                        number_value = to_decimal(cell['text'])
                                        if number_value is not None:
                                            df.iat[r, c]['text'] = str(number_value * 1000000)
                                            unit_million_flag = True
                            break
                else:
                    continue
                break
        unit_million_flags.append(unit_million_flag)

        # flagとunit_million_flagが両方Falseの場合、unit_million_flagsとflagsからpopを開始
        if not flag and not unit_million_flag:
            while unit_million_flags:
                # リストから最後に追加された要素をpopする
                previous_unit_flag = unit_million_flags.pop()  # table_idを指定せず、最後の要素を取り出す
                previous_flag = flags.pop()  # 同様にflagsからも最後の要素を取り出す

                if previous_flag:
                    # previous_flagがTrueなら、即座にループを抜ける
                    break
                elif previous_unit_flag:
                    # previous_unit_flagがTrueなら、全ての数値セルに100万倍を適用
                    for row in range(df.shape[0]):
                        for col in range(df.shape[1]):
                            cell_value = df.iat[row, col]
                            if isinstance(cell_value, dict) and 'text' in cell_value:
                                number_value = to_decimal(cell_value['text'])
                                if number_value is not None:
                                    df.iat[row, col]['text'] = str(number_value * 1000000)

                    # previous_unit_flagがTrueの場合、unit_million_flagsにTrue、flagsにFalseを追加して終了
                    unit_million_flags.append(True)
                    flags.append(False)
                    break
            else:
                # unit_million_flagsとflagsが空の場合、Falseをリストに追加
                unit_million_flags.append(False)
                flags.append(False)

        for row in range(df.shape[0]):
            for col in range(df.shape[1]):
                cell_value = df.iat[row, col]

                if isinstance(cell_value, dict) and 'text' in cell_value:
                    text_value = cell_value['text']

                    # 既存の正規化処理
                    text_value = remove_special_marker(text_value)
                    text_value = remove_bracketed_numbers(text_value)
                    
                    # 新しい正規化処理を適用
                    text_value = normalize_stock_units(text_value)
                    text_value = normalize_date(text_value)
                    text_value = normalize_birth_date(text_value)
                    text_value = normalize_currency(text_value)

                    # 修正後のテキストをセルに反映
                    cell_value['text'] = text_value

    return tables




def process_and_save_tables(root_dir, unit_list_csv, line_list_csv, use_normalize):
    if use_normalize:
        processed_pkl_root = os.path.join(root_dir, '../html2pkltable_normalized')  # 新しい保存先のルートディレクトリ
    else:
        processed_pkl_root = os.path.join(root_dir, '../html2pkltable_origin')  # 新しい保存先のルートディレクトリ
        
    # root_dir内の全てのサブディレクトリを取得
    for file_path in iterate_search_files(root_dir, '.html'):
        tables = extract_tables_from_html(file_path)

        if use_normalize:
            tables = normalize_dataframes(line_list_csv, unit_list_csv, tables)

        # HTMLファイルの親ディレクトリ名（S100~~~~部分）を取得
        subdir_name = os.path.basename(os.path.dirname(file_path))
        # 新しい保存先ディレクトリを作成
        processed_subdir_path = os.path.join(processed_pkl_root, subdir_name)
        os.makedirs(processed_subdir_path, exist_ok=True)
        # 各テーブルをpickle形式で保存
        for table_id, df in tables.items():
            output_file = os.path.join(processed_subdir_path, f"{table_id}.pkl")
            df.to_pickle(output_file)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='HTMLファイルからpklで保存されたtableを生成するプログラム')
    parser.add_argument('--dir', type=str, required=True, help='HTMLファイルが保存されているディレクトリを指定 例: table_retrieval/valid_tr/reports_tr_valid')
    parser.add_argument('--normalize', action='store_true', help='このオプションを使用することで，各セルのテキストが正規化される(TQA向き)')

    args = parser.parse_args()

    unit_list_csv = 'csv/unit_list_top.csv'
    line_list_csv = 'csv/line_list.csv'

    process_and_save_tables(args.dir, unit_list_csv, line_list_csv, args.normalize)