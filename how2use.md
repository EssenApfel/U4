1. html2pkltable.pyを回して正規化されたテーブル(.pkl)を作る(--normalizeオプションで，各テキストの正規化)
python3 html2pkltable.py --dir table_qa/train_tqa/reports_tqa_train --normalize

1.5. tde4u4.pyを回して各セルを分類する(分類することで，各セルにprobabilityが追加される)(for TQA)．
nohup python3 tde4u4.py --dir table_qa/valid_tqa/html2pkltable_normalized --device cuda:1

2. sentencegen.pyを回してテーブルをもとに文章(.csv)を作る(--tdeオプションで，tdeを使用した処理, --averageオプションでmicro平均かmacro平均でスコア計算)
python3 sentencegen.py /home/takasago/U4/table_qa/train_tqa/html2pkltable_normalized/ --tde --average macro --label probability

2.5 get_above_paragraph.pyを回して表上のparagraphを持ってくる(for TR)

3. tqa_bm25.pyを回して文章をもとに正解っぽい文を選ぶ(predicts.json)
python3 table_qa/src/tqa_bm25.py table_qa/test_tqa/sentencegen_output/output_macro_probability/ --datatype test

4. eval_tqa.pyを回して評価する

