from transformers import AutoTokenizer

# トークナイザーの読み込み（例: DeBERTa用）
tokenizer = AutoTokenizer.from_pretrained("ku-nlp/deberta-v3-base-japanese")

# テストデータ
s1 = "これは文1です。"
s2 = "これは文2です。"

# トークナイズ
encoding = tokenizer(
    s1, s2,
    add_special_tokens=True,
    max_length=128,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)

# トークン化された結果
print(encoding["input_ids"])
print(tokenizer.convert_ids_to_tokens(encoding["input_ids"][0]))