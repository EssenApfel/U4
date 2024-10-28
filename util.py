from tqdm import tqdm  # プログレスバー用ライブラリ
import os
import deepl
import time

def iterate_search_files(root_dir, extension):
    """
    ディレクトリを探索し，ファイルを順に返していくイテレータ
    使用例
    for html_file in iterate_search_files(root_dir, '.html'):
        print(html_file)
    """
    # root_dir内の全てのサブディレクトリを
    subdirs = [os.path.join(root_dir, subdir) for subdir in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, subdir))]

    # 全てのサブディレクトリに対して処理
    for subdir_path in tqdm(subdirs, desc="Processing directories"):
        # サブディレクトリ内の全てのHTMLファイルを処理
        html_files = [file for file in os.listdir(subdir_path) if file.endswith(extension)]

        for file in tqdm(html_files, desc=f"Processing files in {os.path.basename(subdir_path)}", leave=False):
            file_path = os.path.join(subdir_path, file)
            yield file_path

