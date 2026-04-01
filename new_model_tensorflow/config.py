# 📁 config.py
# すべての設定を管理するファイル

import os
import datetime

# --- 基本設定 ---
DT_NOW = datetime.datetime.now()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- データ設定 ---
DATA_CONFIG = {
    "learning_data_path": os.path.join(BASE_DIR, "py/learning_data/2025_6_20"),
    "test_data_path": os.path.join(BASE_DIR, "py/test_data/2025_6_20/input"),
    "input_dir_name": "input",
    "label_dir_name": "label",
    "file_pattern": "input_1_*.txt", # 学習データの入力ファイルパターン
}

# --- 前処理・正規化設定 ---
PREPROCESS_CONFIG = {
    "use_db_scale": True,    # dBスケールを使用するか
    "eps": 1e-12,            # log(0)を避けるための微小値
    "min_db": -120.0,        # 正規化に使用する最小dB値
    "max_db": 0.0,           # 正規化に使用する最大dB値
    "num_samples": 1024,     # 1チャープあたりのサンプル数
    "num_chirps": 16,        # チャープ数
}

# --- モデル設定 ---
MODEL_CONFIG = {
    "hidden_size": 200,      # RNNの隠れ層のサイズ
    "num_rnn_layers": 3,     # RNN層の数
    "num_attention_heads": 4,# Attentionのヘッド数
    "dropout_rate": 0.1,
}

# --- 学習設定 ---
TRAIN_CONFIG = {
    "epochs": 100,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "validation_split": 0.2, # 検証データの割合 (20%)
    "model_save_path": os.path.join(BASE_DIR, "saved_models", f"Attention_biLSTM_{DT_NOW:%Y%m%d_%H%M}"),
}

# --- 推論設定 ---
INFERENCE_CONFIG = {
    "trained_model_path": TRAIN_CONFIG["model_save_path"], # 学習済みモデルのパス (学習後に自動設定)
    "output_mat_path": os.path.join(BASE_DIR, "py/test/inference_results", f"{DT_NOW:%Y%m%d_%H%M}"),
}