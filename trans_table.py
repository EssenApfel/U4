from util import iterate_search_files
import time
import deepl
import argparse 
import pandas as pd
import os
import regex as re

# グローバル変数，翻訳される文字数
total_translation_chars = 0

def translator_ja2en(text, deepl_api):
    '''
    与えられたtextを英語に翻訳する関数．
    エラーが出て翻訳できなかった場合は空を返す．
    '''
    translator = deepl.Translator(deepl_api)
    max_retries = 3  # 最大リトライ回数
    retry_count = 0
    if text == "":
        return ""
    while retry_count < max_retries:
        try:
            result = translator.translate_text(
                text, 
                source_lang="JA",  # 日本語から
                target_lang="EN-US",  # 英語へ
            )
            return result.text
        except deepl.DeepLException as e:
            retry_count += 1
            print(f"エラーが発生しました: {e}. テキスト内容: {text}")
            time.sleep(1)  # 少し待ってから再試行
    else:
        print(f"最大リトライ回数 {max_retries} に達しました。処理を中止します。")
        return text

def save_translated_pkl(df, original_pkl_path, dest_root):
    '''
    翻訳後のDataFrameを保存先ディレクトリに保存する関数
    引数:
        df: 翻訳後のDataFrame
        original_pkl_path: 元のpklファイルのパス
        dest_root: 保存先のルートディレクトリ
    '''
    # [docid]を含む相対パスを生成
    relative_path = os.path.relpath(original_pkl_path, start=input_dir)
    # 保存先のpklファイルのパスを生成
    dest_pkl_path = os.path.join(dest_root, relative_path)

    # 保存先のディレクトリを作成（存在しない場合）
    os.makedirs(os.path.dirname(dest_pkl_path), exist_ok=True)

    # 翻訳後のDataFrameをpklファイルとして保存
    df.to_pickle(dest_pkl_path)


def is_translation_needed(text):
    '''
    翻訳が必要かどうかを判定する関数。
    - 数字や英字のみの場合は翻訳不要とする。
    - その他の日本語や特殊文字が含まれる場合は翻訳が必要。

    引数:
        text: チェックするテキスト
    返り値:
        bool: 翻訳が必要かどうか
    '''
    # 半角英数字と記号のみで構成されている場合は翻訳不要
    if re.fullmatch(r'[\w\s\p{P}]*', text):
        return False
    return True

def translate_pkl_file(pkl_path, deepl_api):
    '''
    指定されたpklファイルのDataFrameの各セルのテキストを英語に翻訳する関数．
    引数:
        pkl_path: pklファイルのパス
        deepl_api: DeepL APIキー
    返り値:
        翻訳後のDataFrame
    '''
    global total_translation_chars  # グローバル変数を使用

    # pklファイルを読み込む
    df = pd.read_pickle(pkl_path)

    # 各セルを翻訳
    for row in range(df.shape[0]):
        for col in range(df.shape[1]):
            cell = df.iloc[row, col]
            if isinstance(cell, dict) and 'text' in cell:
                # セルのテキストを取得
                text = cell['text']
                # 翻訳の必要があるかを確認
                if is_translation_needed(text):
                    # 翻訳する文字数をカウント
                    total_translation_chars += len(text)
                    # 翻訳を実行
                    translated_text = translator_ja2en(text, deepl_api)
                    # 翻訳後のテキストを保存
                    cell['translated'] = translated_text
                # else:
                    # 翻訳が不要な場合、元のテキストをそのまま保存
                cell['translated'] = text

    return df

if __name__ == "__main__":

    # argparseを使ってコマンドライン引数を取得
    parser = argparse.ArgumentParser(description="pklファイルを読み込み，翻訳して返すプログラム")
    parser.add_argument('--input_dir', type=str, required=True, help='元pklファイルのrootディレクトリの指定')
    parser.add_argument('--output_dir', type=str, required=True, help='翻訳後のpklファイルの出力先')
    args = parser.parse_args()

    # 取得した引数を変数に代入
    input_dir = args.input_dir
    output_dir = args.output_dir
    root_path = input_dir
    dest_root = output_dir


    deepl_api = "df670272-84f9-4878-a9a2-445a733a8218:fx"   #takasagosou


    # pklファイルを順に翻訳
    for pkl_path in iterate_search_files(input_dir, ".pkl"):
        translated_df = translate_pkl_file(pkl_path, deepl_api)
        save_translated_pkl(translated_df, pkl_path, output_dir)
    print(total_translation_chars)
