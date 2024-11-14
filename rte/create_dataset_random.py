import json
import os
import pandas as pd
import random
from tqdm import tqdm

# データセットが偏るのを防止
# labelが0のものを集めて，ランダムで一つだけデータセットに加える．

def create_training_data(questions_path, csv_root, output_path):
    # 出力用のリストを定義
    labels = []
    questions = []
    sentences = []

    # questions.json を読み込む
    with open(questions_path, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)
    
    # プログレスバーを表示しながら全ての question_train を処理
    for key, question_info in tqdm(questions_data.items(), desc="Processing questions"):
        question = question_info['question']
        doc_id = question_info['doc_id']
        table_id = question_info['table_id']
        target_cell_id = question_info['cell_id']
        
        # 該当するCSVファイルを読み込む
        csv_path = os.path.join(csv_root, doc_id, f"{table_id}_sentences.csv")
        if os.path.exists(csv_path):
            csv_data = pd.read_csv(csv_path)
            
            # labelが0となる文のリストを作成
            negative_sentences = []
            
            for _, row in csv_data.iterrows():
                sentence = row['sentence']
                cell_id = row['id']  # CSVの'id'列はcell_idを示す

                # cell_id の一致を確認し、ラベルを設定
                if cell_id == target_cell_id:
                    # label 1 の場合
                    labels.append(1)
                    questions.append(question)
                    sentences.append(sentence)
                else:
                    # label 0 の場合はnegative_sentencesに追加
                    negative_sentences.append(sentence)
            
            # label 0 の文をランダムに1つ選択して追加
            if negative_sentences:
                random_sentence = random.choice(negative_sentences)
                labels.append(0)
                questions.append(question)
                sentences.append(random_sentence)
    
    # 出力データフレームを作成
    output_df = pd.DataFrame({
        'label': labels,
        'question': questions,
        'sentence': sentences
    })
    
    # 重複する行を削除
    output_df = output_df.drop_duplicates()
    
    # TSVファイルとして保存
    output_df.to_csv(output_path, sep='\t', index=False, encoding='utf-8')
    print(f"Training data saved to {output_path}")

# 実行例
questions_path = 'table_qa/train_tqa/questions_tqa_train.json'
csv_root = 'table_qa/train_tqa/sentencegen_output/null_noappend/'
output_path = 'training_data_random.tsv'
create_training_data(questions_path, csv_root, output_path)