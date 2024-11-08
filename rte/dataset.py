import pandas as pd

# ファイルパスを指定してTSVファイルを読み込む
df = pd.read_csv("jrte-corpus/data/rte.lrec2020_sem_long.tsv", sep='\t', header=None, usecols=[1, 2, 3])

# 列の名前をわかりやすく設定
df.columns = ["label", "s1", "s2"]

# データフレームを表示
print(df.head())