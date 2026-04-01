# 📁 utils.py
# 共通のヘルパー関数を定義するファイル

import numpy as np
import scipy.io

def load_data_from_txt(filepath):
    """テキストファイルからデータを読み込み、NumPy配列として返す"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    data = [list(map(float, line.strip().split())) for line in lines]
    return np.array(data)

def to_db_scale(linear_data, eps=1e-12):
    """線形スケールのパワーデータをdBスケールに変換する"""
    power_linear = linear_data**2
    return 10 * np.log10(power_linear + eps)

def normalize_db_data(db_data, min_db, max_db):
    """dBスケールのデータを0-1の範囲に正規化する"""
    normalized = (db_data - min_db) / (max_db - min_db)
    return np.clip(normalized, 0, 1) # 範囲外の値が出ないようにクリッピング

def denormalize_db_data(normalized_data, min_db, max_db):
    """0-1に正規化されたデータを元のdBスケールに戻す"""
    return normalized_data * (max_db - min_db) + min_db

def save_as_mat(filepath, data_dict):
    """データを.matファイルとして保存する"""
    scipy.io.savemat(filepath, data_dict)
    print(f"ファイルを保存しました: {filepath}")