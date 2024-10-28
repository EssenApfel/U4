import json
from sentencegen import find_best_split
from util import iterate_search_files
import pandas as pd
import re
import os

def extract_tab_number(table_id):
    """table-idからtab番号を抽出する。"""
    match = re.search(r'-tab(\d+)', table_id)
    return int(match.group(1)) if match else float('inf')

def evaluate_splits(input_pkl_path, output_json_path):
    # 結果を格納する辞書
    results = {}

    # .pklファイルを再帰的に探索
    for file_path in iterate_search_files(input_pkl_path, ".pkl"): 
        docid = os.path.basename(os.path.dirname(file_path))  # ディレクトリ名を取得（例: S100ILF5）
        table_id = os.path.splitext(os.path.basename(file_path))[0]  # ファイル名から拡張子を除去（例: S100ILF5-0000000-tab1）

        # resultsにdocidが存在しない場合、新たに辞書を作成
        if docid not in results:
            results[docid] = {}

        # DataFrameを読み込む
        df = pd.read_pickle(file_path)

        # 各パターンで分割を実行
        split_argmax_micro = find_best_split(df, label_type='argmax', average_type='micro')
        split_argmax_macro = find_best_split(df, label_type='argmax', average_type='macro')
        split_prob_micro = find_best_split(df, label_type='probability', average_type='micro')
        split_prob_macro = find_best_split(df, label_type='probability', average_type='macro')

        # 結果を辞書に格納
        results[docid][table_id] = {
            "argmax_micro": {"row_split": split_argmax_micro[0], "col_split": split_argmax_micro[1]},
            "argmax_macro": {"row_split": split_argmax_macro[0], "col_split": split_argmax_macro[1]},
            "probability_micro": {"row_split": split_prob_micro[0], "col_split": split_prob_micro[1]},
            "probability_macro": {"row_split": split_prob_macro[0], "col_split": split_prob_macro[1]}
        }

    # 結果をtab番号でソート
    sorted_results = {
        docid: dict(sorted(table_ids.items(), key=lambda item: extract_tab_number(item[0]))) 
        for docid, table_ids in sorted(results.items())
    }

    # 結果をJSONファイルに保存
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(sorted_results, f, ensure_ascii=False, indent=4)

    print(f"結果が {output_json_path} に保存されました。")

# 使用例
pkl_root_path = 'table_qa/valid_tqa/html2pkltable/update_date'
output_json = 'predict_splitpoint_10_updatedate.json'
evaluate_splits(pkl_root_path, output_json)