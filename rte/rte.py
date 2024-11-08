import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


class CustomDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        label = row['label']
        s1 = row['s1']
        s2 = row['s2']
        
        encoding = self.tokenizer(
            s1, s2,                       # s1とs2を別々の引数として渡す
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
        return item
    

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    


# ファイルパスを指定してTSVファイルを読み込む
df = pd.read_csv("jrte-corpus/data/rte.lrec2020_sem_long.tsv", sep='\t', header=None, usecols=[1, 2, 3])

# 列の名前をわかりやすく設定
df.columns = ["label", "s1", "s2"]


# モデル指定
model_name = "cl-tohoku/bert-large-japanese-v2"
    
tokenizer = BertTokenizer.from_pretrained(model_name)
train_dataset = CustomDataset(df, tokenizer, max_len=128)

train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = CustomDataset(train_df, tokenizer, max_len=128)
eval_dataset = CustomDataset(eval_df, tokenizer, max_len=128)

model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)


training_args = TrainingArguments(
    output_dir='./results',           # 出力ディレクトリ
    num_train_epochs=3,               # エポック数
    per_device_train_batch_size=16,   # バッチサイズ
    per_device_eval_batch_size=16,    # 評価時のバッチサイズ
    warmup_steps=500,                 # 学習率のウォームアップステップ数
    weight_decay=0.01,                # 重みの減衰率
    logging_dir='./logs',             # ログの出力ディレクトリ
    logging_steps=10,
    evaluation_strategy="epoch"       # 各エポック終了後に評価
)


trainer = Trainer(
    model=model,                         # モデル
    args=training_args,                  # 訓練設定
    train_dataset=train_dataset,         # トレーニングデータセット
    eval_dataset=eval_dataset,           # 評価データセット
    compute_metrics=compute_metrics      # 精度計算関数
)
trainer.train()

eval_results = trainer.evaluate()
print(eval_results)