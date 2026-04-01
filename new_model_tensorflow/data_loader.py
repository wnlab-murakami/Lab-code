# 📁 data_loader.py
# データの読み込み、前処理、データセットの作成を担当

import tensorflow as tf
import numpy as np
import glob
from natsort import natsorted
import os

# 設定ファイルとユーティリティファイルをインポート
import config
import utils

def get_datasets(batch_size, validation_split):
    """学習用と検証用のtf.data.Datasetを返す"""
    
    # 設定からパスを取得
    input_path = os.path.join(config.DATA_CONFIG["learning_data_path"], config.DATA_CONFIG["input_dir_name"])
    label_path = os.path.join(config.DATA_CONFIG["learning_data_path"], config.DATA_CONFIG["label_dir_name"])
    
    input_files = natsorted(glob.glob(os.path.join(input_path, config.DATA_CONFIG["file_pattern"])))
    label_files = natsorted(glob.glob(os.path.join(label_path, config.DATA_CONFIG["file_pattern"].replace("input", "label"))))

    if not input_files or not label_files:
        raise FileNotFoundError("学習データが見つかりませんでした。config.pyのパスを確認してください。")

    # 全てのデータを読み込む
    all_inputs, all_labels = [], []
    print("データを読み込んでいます...")
    for i_file, l_file in zip(input_files, label_files):
        # 各ファイルは (チャープ数 x サンプル数) のデータ
        input_data = utils.load_data_from_txt(i_file)
        label_data = utils.load_data_from_txt(l_file)
        
        # モデルの入力形式 (サンプル数, チャープ数) に合わせるため転置
        all_inputs.append(input_data.T)
        all_labels.append(label_data.T)
    
    # NumPy配列に変換
    all_inputs = np.array(all_inputs, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.float32)

    print(f"データ読み込み完了: {all_inputs.shape[0]} 個のデータ")

    # 前処理（dB変換と正規化）
    if config.PREPROCESS_CONFIG["use_db_scale"]:
        print("データをdBスケールに変換し、正規化します...")
        all_inputs = utils.to_db_scale(all_inputs, eps=config.PREPROCESS_CONFIG["eps"])
        all_inputs = utils.normalize_db_data(all_inputs, config.PREPROCESS_CONFIG["min_db"], config.PREPROCESS_CONFIG["max_db"])
        
        all_labels = utils.to_db_scale(all_labels, eps=config.PREPROCESS_CONFIG["eps"])
        all_labels = utils.normalize_db_data(all_labels, config.PREPROCESS_CONFIG["min_db"], config.PREPROCESS_CONFIG["max_db"])

    # 学習データと検証データに分割
    num_data = all_inputs.shape[0]
    num_validation = int(num_data * validation_split)
    
    # データをシャッフル
    indices = np.arange(num_data)
    np.random.shuffle(indices)
    all_inputs = all_inputs[indices]
    all_labels = all_labels[indices]
    
    train_inputs, valid_inputs = all_inputs[:-num_validation], all_inputs[-num_validation:]
    train_labels, valid_labels = all_labels[:-num_validation], all_labels[-num_validation:]

    # tf.data.Datasetを作成
    def create_dataset(inputs, labels):
        dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
        dataset = dataset.shuffle(buffer_size=1024)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

    train_dataset = create_dataset(train_inputs, train_labels)
    valid_dataset = create_dataset(valid_inputs, valid_labels)

    print(f"学習データ: {len(train_inputs)} 個, 検証データ: {len(valid_inputs)} 個")
    return train_dataset, valid_dataset