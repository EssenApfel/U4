import json

# ファイルの読み込み
with open('split_gold.json', 'r', encoding='utf-8') as f:
    split_data = json.load(f)

with open('table_qa/valid_tqa/questions_tqa_valid.json', 'r', encoding='utf-8') as f:
    questions_data = json.load(f)

# すべてのis_includeをfalseに初期化
for doc_id, tables in split_data.items():
    for table_id in tables:
        split_data[doc_id][table_id]["is_include"] = False

# questions_dataを基にtrueに更新
for question_id, question_info in questions_data.items():
    table_id = question_info["table_id"]
    doc_id = question_info["doc_id"]
    
    # doc_idがsplit_dataに存在しない場合は新しく辞書を作成
    if doc_id not in split_data:
        split_data[doc_id] = {}

    # table_idが存在する場合はis_includeをtrueに設定
    if table_id in split_data[doc_id]:
        split_data[doc_id][table_id]["is_include"] = True

# 結果の保存
with open('split_gold_updated.json', 'w', encoding='utf-8') as f:
    json.dump(split_data, f, ensure_ascii=False, indent=4)

print("処理が完了しました。'split_gold_updated.json'に結果が保存されています。")