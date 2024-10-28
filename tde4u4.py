import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm  # プログレスバーのためのライブラリ
from util import iterate_search_files
import argparse

# predict関数
def predict(text, tokenizer, model, device):
    inputs = tokenizer(text, add_special_tokens=True, return_tensors="pt", padding='max_length', max_length=512).to(device)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    if input_ids.size(1) > 512:
        input_ids = input_ids[:, :512]
        attention_mask = attention_mask[:, :512]

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    ps = torch.nn.Softmax(dim=1)(outputs.logits)

    # result = torch.argmax(ps).item()
    probabilities = ps.detach().cpu().numpy().tolist()[0]
    return probabilities

def process_pkl_file(pkl_file_path, tokenizer, model, device):
    # .pklファイルを読み込む
    df = pd.read_pickle(pkl_file_path)

    # DataFrameの各セルに対して処理を行う
    for index, row in df.iterrows():
        for col in df.columns:
            cell = df.at[index, col]

            # セル内が辞書型であり、'text'キーが存在する場合
            if isinstance(cell, dict) and 'text' in cell:
                text = cell['text']
                
                # predict関数で確率を予測
                probabilities = predict(text, tokenizer, model, device)
                
                # 確率リストを'type'に格納
                cell['type'] = probabilities
                
                # 更新した辞書をセルに戻す
                df.at[index, col] = cell

    # 上書き保存
    df.to_pickle(pkl_file_path)
    # print(f"Updated DataFrame saved to {pkl_file_path}")

def main():
    # モデルとトークナイザの設定
    data_path = '/home/takasago/TDE/data/'
    model_n = 'model_2e-5_32_20_0.1'
    model_path = data_path + model_n
    model_name = "cl-tohoku/bert-large-japanese-v2"

    # ラベルマッピング
    # label_mapping = {
    #     0: 'metadata',
    #     1: 'header',
    #     2: 'attribute',
    #     3: 'data'
    # }

    parser = argparse.ArgumentParser(description='pklファイルで保存されたTable内のdataをTDEモデルで分類するプログラム')
    parser.add_argument('--dir', type=str, required=True, help='pklファイルが保存されているディレクトリを指定 例: table_qa/valid_tqa/html2pkltable_normalized')
    parser.add_argument('--device', type=str, required=True, help='BERTを動かすデバイスの指定．例: cuda:0')
    args = parser.parse_args()

    # 使用例: ルートディレクトリのパスを指定して処理


    # トークナイザとモデルのロード
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(args.device)


    for file_path in iterate_search_files(args.dir, '.pkl'):
        process_pkl_file(file_path, tokenizer, model, args.device)

if __name__ == '__main__':
    main()