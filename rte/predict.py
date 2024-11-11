import torch
from transformers import BertJapaneseTokenizer, BertForSequenceClassification

# モデルとトークナイザーのロード
model_name_or_path = './fine_tuned_model'
tokenizer = BertJapaneseTokenizer.from_pretrained(model_name_or_path)
model = BertForSequenceClassification.from_pretrained(model_name_or_path)

# モデルを評価モードに変更し、GPUがあれば移動
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def predict(s1, s2):
    # 入力テキストをトークナイズ
    encoding = tokenizer(
        s1, s2,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    # デバイスへ移動
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # 推論
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    # ログitsから予測ラベルを取得
    logits = outputs.logits
    # predicted_class = torch.argmax(logits, dim=1).item()

    import torch.nn.functional as F
    probs = F.softmax(logits, dim=1)
    predicted_class = probs.squeeze().tolist()
    
    return predicted_class

# サンプル予測
s1 = "このお風呂は大きいですね"  # 例としての文1
s2 = "お腹が減りました"  # 例としての文2
prediction = predict(s1, s2)
print("予測ラベル:", prediction)
