from sentence_transformers import SentenceTransformer, util

device = "cuda:0"
# SentenceBERTモデルをロードし、CUDAデバイスに転送
model = SentenceTransformer('sonoisa/sentence-bert-base-ja-mean-tokens-v2')
model = model.to(device)  # CUDAデバイスに転送

# クエリと文書をエンコード
query = "インドカレー屋で提供されているラッシーは，120円だ．"
docs = [
    "ラッシーは，150円だ．",
    "ラッシーは，120円だ．",
    "カレーが好きだ。中でも、インドカレーが一番好きだ。",
    "自宅で作ったラッシーも美味しい。",
    "欧風カレーとインドカレーは全くの別物だが、どちらも美味しい。",
    "インドカレーが好きだ。"
]
docsid = [1, 2, 3, 4, 5, 6]
docsvalue = ["a", "b", "c", "d", "e", "f"]
# GPUでエンコード処理
query_embedding = model.encode(query, device=device)
doc_embeddings = model.encode(docs, device=device)

# クエリと各文書のコサイン類似度を計算
similarities = util.cos_sim(query_embedding, doc_embeddings)[0]

# 類似度の高い順にソートして表示
sorted_indices = similarities.argsort(descending=True)
sorted_docs = [docs[i] for i in sorted_indices]
sorted_docsid = [docsid[i] for i in sorted_indices]
sorted_docsvalue = [docsvalue[i] for i in sorted_indices]

sorted_similarities = [similarities[i].item() for i in sorted_indices]
# 結果を表示
for doc, similarity, docid, docvalue in zip(sorted_docs, sorted_similarities, sorted_docsid, sorted_docsvalue):
    print(f"文書: {doc}, 類似度: {similarity:.4f}, ID: {docid}, Value: {docvalue}")