import os
import pandas as pd
import re
import numpy as np
import argparse
from tqdm import tqdm
from util import iterate_search_files

def is_dict_and_not_empty(cell):
    return isinstance(cell, dict) and cell.get('text', '') != ''

# 1023に指摘された「textが空文字ならば信頼性がない」と言われたため，空なら無視
# 1023に指摘された「空文字を軸にテキストを生成するのは意味がない」と言われたため，からなら無視
def calculate_score_argmax(table, row_split, col_split, average_type='micro'):
    upper = table.iloc[:row_split, col_split:]
    left = table.iloc[row_split:, :col_split]
    upper_left = table.iloc[:row_split, :col_split]
    lower_right = table.iloc[row_split:, col_split:]
        
    non_data_upper_left_count = sum(
        1 for cell in upper_left.to_numpy().flatten() 
        if is_dict_and_not_empty(cell) and np.argmax(cell.get('type', [0, 0, 0, 0])) != 3
    )
    total_upper_left_cells = sum(
        1 for cell in upper_left.to_numpy().flatten() 
        if is_dict_and_not_empty(cell)
    )

    non_data_upper_count = sum(
        1 for cell in upper.to_numpy().flatten() 
        if is_dict_and_not_empty(cell) and np.argmax(cell.get('type', [0, 0, 0, 0])) != 3
    )
    total_upper_cells = sum(
        1 for cell in upper.to_numpy().flatten() 
        if is_dict_and_not_empty(cell)
    )

    non_data_left_count = sum(
        1 for cell in left.to_numpy().flatten() 
        if is_dict_and_not_empty(cell) and np.argmax(cell.get('type', [0, 0, 0, 0])) != 3
    )
    total_left_cells = sum(
        1 for cell in left.to_numpy().flatten() 
        if is_dict_and_not_empty(cell)
    )

    data_lower_right_count = sum(
        1 for cell in lower_right.to_numpy().flatten() 
        if is_dict_and_not_empty(cell) and np.argmax(cell.get('type', [0, 0, 0, 0])) == 3
    )
    total_lower_right_cells = sum(
        1 for cell in lower_right.to_numpy().flatten() 
        if is_dict_and_not_empty(cell)
    )

    non_data_total_correct = non_data_left_count + non_data_upper_count + non_data_upper_left_count
    non_data_total_cells = total_upper_cells + total_upper_left_cells + total_left_cells

    data_total_correct = data_lower_right_count
    data_total_cells = total_lower_right_cells

    if non_data_total_cells == 0 or data_total_cells == 0:
        return 0  # Return zero if any denominator is zero

    if average_type == 'macro':
        score = ((non_data_total_correct / non_data_total_cells) + (data_total_correct / data_total_cells)) / 2
    else:
        score = (non_data_total_correct + data_total_correct) / (non_data_total_cells + data_total_cells)

    return score

def calculate_score_probability(table, row_split, col_split, average_type='micro'):
    upper = table.iloc[:row_split, col_split:]
    left = table.iloc[row_split:, :col_split]
    upper_left = table.iloc[:row_split, :col_split]
    lower_right = table.iloc[row_split:, col_split:]
        
    non_data_upper_left_correct = sum(
        sum(cell['type'][:3]) for cell in upper_left.to_numpy().flatten() 
        if is_dict_and_not_empty(cell) and 'type' in cell
    )
    total_upper_left_cells = sum(
        1 for cell in upper_left.to_numpy().flatten() 
        if is_dict_and_not_empty(cell)
    )

    non_data_upper_correct = sum(
        sum(cell['type'][:3]) for cell in upper.to_numpy().flatten() 
        if is_dict_and_not_empty(cell) and 'type' in cell
    )
    total_upper_cells = sum(
        1 for cell in upper.to_numpy().flatten() 
        if is_dict_and_not_empty(cell)
    )

    non_data_left_correct = sum(
        sum(cell['type'][:3]) for cell in left.to_numpy().flatten() 
        if is_dict_and_not_empty(cell) and 'type' in cell
    )
    total_left_cells = sum(
        1 for cell in left.to_numpy().flatten() 
        if is_dict_and_not_empty(cell)
    )

    data_lower_right_correct = sum(
        cell['type'][3] for cell in lower_right.to_numpy().flatten() 
        if is_dict_and_not_empty(cell) and 'type' in cell
    )
    total_lower_right_cells = sum(
        1 for cell in lower_right.to_numpy().flatten() 
        if is_dict_and_not_empty(cell)
    )

    non_data_total_correct = non_data_left_correct + non_data_upper_correct + non_data_upper_left_correct
    non_data_total_cells = total_upper_cells + total_upper_left_cells + total_left_cells

    data_total_correct = data_lower_right_correct
    data_total_cells = total_lower_right_cells

    if non_data_total_cells == 0 or data_total_cells == 0:
        return 0  # Return zero if any denominator is zero

    if average_type == 'macro':
        score = ((non_data_total_correct / non_data_total_cells) + (data_total_correct / data_total_cells)) / 2
    else:
        score = (non_data_total_correct + data_total_correct) / (non_data_total_cells + data_total_cells)

    return score

def find_best_split(table, average_type='micro', label_type='argmax'):
    max_score = -1
    best_split = (0, 0)
    # 1021追加, 10までrangeを伸ばすことで精度向上
    tab_range = 10 # セルを分割する最大の大きさ

    for row_split in range(1, min(tab_range+1, table.shape[0])):
        for col_split in range(1, min(tab_range+1, table.shape[1])):
            if label_type=='argmax':
                score = calculate_score_argmax(table, row_split, col_split, average_type)
            else:
                score = calculate_score_probability(table, row_split, col_split, average_type)
            if score > max_score:
                max_score = score
                best_split = (row_split, col_split)
    
    return best_split

def generate_sentences_from_table(table, tde_processed=False, average_type='micro', label_type='argmax'):
    # tableをもとにsentences, cell_ids, datasを得る関数

    # tdeの処理を事前にしていれば，分割位置を探索
    if tde_processed:
        split_row, split_col = find_best_split(table, average_type, label_type)
    else:
        # していなければ，分割は固定m
        split_row, split_col = 1, 1
    
    sentences = []
    cell_ids = []
    datas = []
    
    # データセル部分のループ (split_row, split_col 以降がdataセル部分)
    for row in range(split_row, table.shape[0]):
        for col in range(split_col, table.shape[1]):
            data_cell = table.iat[row, col]
            
            # データセルが辞書型でない場合はスキップ
            if not isinstance(data_cell, dict):
                continue

            # タイトルセルを取得
            title_cells_row = [table.iat[i, col] for i in range(split_row)]
            title_cells_col = [table.iat[row, j] for j in range(split_col)]
            
            # タイトルセルが辞書型の場合のみ'text'を取り出す
            title_text_row = re.sub(r'^、+', '', "、".join(
                set([cell['text'] for cell in title_cells_row if isinstance(cell, dict) and 'text' in cell])
            ))
            title_text_col = re.sub(r'^、+', '', "、".join(
                set([cell['text'] for cell in title_cells_col if isinstance(cell, dict) and 'text' in cell])
            ))
            
            # 文章を生成
            if not title_text_col:
                sentence = f"{title_text_row}は{data_cell.get('text', '')}です。"
            elif not title_text_row:
                sentence = f"{title_text_col}は{data_cell.get('text', '')}です。"
            else:
                sentence = f"{title_text_row}の{title_text_col}は{data_cell.get('text', '')}です。"
            
            if data_cell.get('text', '') != '':
                sentences.append(sentence)
                # 使用したセルの'id'を追加
                cell_ids.append(data_cell.get('id', ''))
                datas.append(data_cell.get('text', ''))
    
    return sentences, cell_ids, datas

# 使用例
import os
import pandas as pd
import argparse

def process_and_save_sentences(root_dir, tde_processed, average_type, label_type):
    # 保存先ディレクトリ名にaverage_typeとlabel_typeを含める
    save_dir = os.path.join(root_dir, f"../sentencegen_output/output_{average_type}_{label_type}")
    
    for file_path in iterate_search_files(root_dir, '.pkl'):  # pklファイルを探索
        try:
            table = pd.read_pickle(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        # [docid]のディレクトリを取得して保存先を構築
        relative_path = os.path.relpath(file_path, root_dir)
        docid_dir = os.path.dirname(relative_path)
        save_subdir = os.path.join(save_dir, docid_dir)
        os.makedirs(save_subdir, exist_ok=True)

        # 文の生成と保存
        try:
            sentences, cell_ids, datas = generate_sentences_from_table(table, tde_processed, average_type, label_type)

            # 保存用のCSVファイル名を作成
            csv_filename = os.path.join(save_subdir, os.path.splitext(os.path.basename(file_path))[0] + "_sentences.csv")

            # データフレームを作成
            table_output = pd.DataFrame({
                'sentence': sentences,
                'id': cell_ids,
                'data': datas
            })
            # 重複削除
            table_output = table_output.drop_duplicates(subset=['sentence', 'id', 'data'])
            # CSVファイルとして保存
            table_output.to_csv(csv_filename, index=False, encoding='utf-8')

        except Exception as e:
            print(f"Error processing {file_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process .pkl files and generate sentences.")
    parser.add_argument('root_directory_path', type=str, help='Root directory path for .pkl files.')
    parser.add_argument('--tde', action='store_true', help='Flag indicating whether tde processing was done.')
    parser.add_argument('--average', choices=['micro', 'macro'], default='micro', help='Specify average type: micro or macro.')
    parser.add_argument('--label', choices=['argmax', 'probability'], default="argmax", help='TDEによって分類されたLabelをどのように扱ってスコア計算を行うか．argmax, probability')
    args = parser.parse_args()

    root_dir = args.root_directory_path
    tde_processed = args.tde
    average_type = args.average  # 'micro' or 'macro' based on user input
    label_type = args.label

    process_and_save_sentences(root_dir, tde_processed, average_type, label_type)