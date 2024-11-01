def calculate_accuracy(manual_data, model_data):
    # Models to evaluate
    model_names = ["argmax_micro", "argmax_macro", "probability_micro", "probability_macro"]
    # Initialize counters for correct predictions and total count
    correct_counts = {model: 0 for model in model_names}
    total_counts = 0

    # Iterate over each problem set in the manual data
    for problem_set_id, problems in manual_data.items():
        if problem_set_id not in model_data:
            continue  # Skip if model data does not contain this problem set
        
        for problem_id, manual_labels in problems.items():
            is_include = manual_labels.get("is_include", False)
            if not is_include:
                continue

            manual_row = manual_labels["manual"].get("row_split", "")
            manual_col = manual_labels["manual"].get("col_split", "")
            if manual_row == "" or manual_col == "":
                continue  # Skip problems without manual labels

            # Convert manual labels to integers for comparison
            manual_row = int(manual_row)
            manual_col = int(manual_col)
            total_counts += 1

            # Check each model's prediction against the manual labels
            if problem_id not in model_data[problem_set_id]:
                continue  # Skip if the model predictions are not available for this problem

            for model in model_names:
                if model not in model_data[problem_set_id][problem_id]:
                    continue  # Skip if the model predictions are not available

                model_row = model_data[problem_set_id][problem_id][model].get("row_split", "")
                model_col = model_data[problem_set_id][problem_id][model].get("col_split", "")
                if model_row == manual_row and model_col == manual_col:
                    correct_counts[model] += 1

    # Calculate accuracies for each model
    accuracies = {model: correct / total_counts if total_counts > 0 else 0 for model, correct in correct_counts.items()}
    return accuracies, total_counts

import json

# コマンドライン引数の設定
import argparse
import json
parser = argparse.ArgumentParser(
    description="手動ラベルデータとモデル予測データを比較し、精度を算出します。"
)
parser.add_argument(
    '-f', '--file', required=True, help='モデル予測データの JSON ファイルのパスを指定します。'
)
parser.add_argument(
    '-g', '--gold', required=True, help='手動ラベルデータの JSON ファイルのパスを指定します。'
)

# 引数をパース
args = parser.parse_args()
model_json_path = args.file
manual_json_path = args.gold

# 手動ラベルデータの読み込み
with open(manual_json_path, 'r', encoding='utf-8') as f:
    manual_data = json.load(f)

# モデル予測データの読み込み
with open(model_json_path, 'r', encoding='utf-8') as f:
    model_data = json.load(f)

# 精度計算
accuracies, total_counts = calculate_accuracy(manual_data, model_data)
print("精度:", accuracies)
print(f"対象データ数: {total_counts}")