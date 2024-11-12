import pandas as pd
import torch
from transformers import BertJapaneseTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 使用するGPUを1に指定


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
# df = pd.read_csv("jrte-corpus/data/rte.lrec2020_sem_long.tsv", sep='\t', header=None, usecols=[1, 2, 3])
df = pd.read_csv("training_data.tsv", sep='\t', header=None, usecols=[0, 1, 2])
# 列の名前をわかりやすく設定
df.columns = ["label", "s1", "s2"]


# モデル指定
model_name = "cl-tohoku/bert-large-japanese-v2"
    
tokenizer = BertJapaneseTokenizer.from_pretrained(model_name)

train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = CustomDataset(train_df, tokenizer, max_len=256)
eval_dataset = CustomDataset(eval_df, tokenizer, max_len=256)

model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)


from transformers import EarlyStoppingCallback

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,                  # 最大エポック数を30に設定
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="epoch",                # 各エポック終了後に評価
    save_strategy="epoch",                # 各エポック終了後にモデルを保存
    load_best_model_at_end=True           # 早期終了後に最良モデルをロード
)

# Early Stoppingのコールバックを設定
# patience=3 は、3エポック評価が改善されない場合に学習を停止
early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=3)

# Trainerの初期化にcallbackを追加
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    callbacks=[early_stopping_callback]    # EarlyStoppingをコールバックに追加
)
trainer.train()

eval_results = trainer.evaluate()
print(eval_results)

# モデルの保存
trainer.save_model('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')