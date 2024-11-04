import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

path_split_gold = "split_gold.json"
with open(path_split_gold, mode="r") as f:
    data = json.load(f)

# row_splitとcol_splitの組み合わせをカウント
count_dict = {}
for _, tables in data.items():
    for _, info in tables.items():
        is_include = info.get("is_include", False)
        if not is_include:
            continue
        row_split = info["manual"]["row_split"]
        col_split = info["manual"]["col_split"]
        if row_split == "" or col_split == "":
            continue

        count_dict[(row_split, col_split)] = count_dict.get((row_split, col_split), 0) + 1

# カウント結果を行列に変換
max_row_split = max(k[0] for k in count_dict.keys()) + 1
max_col_split = max(k[1] for k in count_dict.keys()) + 1
heatmap_data = np.zeros((max_row_split, max_col_split))

for (row, col), count in count_dict.items():
    heatmap_data[row, col] = count

plt.figure(figsize=(12, 12))
sns.heatmap(heatmap_data, annot=True, fmt="g", cmap="coolwarm", cbar=False)

# メモリとラベルの位置設定
plt.xlabel("")  # 下側のラベルを消す
plt.ylabel("row_split")
plt.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
plt.gca().xaxis.set_label_position('top')
plt.gca().set_xlabel("col_split")

# タイトルは非表示
plt.show()