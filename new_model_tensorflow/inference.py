# 📁 inference.py
# 学習済みモデルを使って推論を実行するスクリプト

import tensorflow as tf
from tensorflow import keras
import numpy as np
import glob
from natsort import natsorted
import os

# 各モジュールをインポート
import config
import utils

def main():
    # 1. 学習済みモデルのロード
    model_path = os.path.join(config.INFERENCE_CONFIG["trained_model_path"], "best_model.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"学習済みモデルが見つかりません: {model_path}")
        
    print(f"モデルをロードしています: {model_path}")
    model = keras.models.load_model(model_path, custom_objects={"AttentionBiLSTM": model.AttentionBiLSTM})
    model.summary()

    # 2. 推論用データの検索
    test_files = natsorted(glob.glob(os.path.join(config.DATA_CONFIG["test_data_path"], "*.txt")))
    if not test_files:
        raise FileNotFoundError(f"テストデータが見つかりません: {config.DATA_CONFIG['test_data_path']}")
    
    # 3. 出力ディレクトリの作成
    os.makedirs(config.INFERENCE_CONFIG["output_mat_path"], exist_ok=True)

    # 4. 各テストデータに対して推論を実行
    for test_file in test_files:
        print(f"\n処理中のファイル: {os.path.basename(test_file)}")
        
        # データを読み込み、(サンプル数, チャープ数)に転置
        input_data = utils.load_data_from_txt(test_file).T.astype(np.float32)
        
        # 前処理 (学習時と全く同じ処理を適用)
        if config.PREPROCESS_CONFIG["use_db_scale"]:
            input_data_db = utils.to_db_scale(input_data, config.PREPROCESS_CONFIG["eps"])
            processed_input = utils.normalize_db_data(input_data_db, config.PREPROCESS_CONFIG["min_db"], config.PREPROCESS_CONFIG["max_db"])
        else:
            # 他の正規化処理をここに記述
            processed_input = input_data # (例：線形スケールの正規化)

        # モデルに入力するためにバッチ次元を追加 (samples, chirps) -> (1, samples, chirps)
        model_input = np.expand_dims(processed_input, axis=0)

        # 推論実行
        predicted_output_normalized = model.predict(model_input)
        
        # バッチ次元を削除
        predicted_output_normalized = np.squeeze(predicted_output_normalized, axis=0)

        # 後処理 (元のスケールに戻す)
        if config.PREPROCESS_CONFIG["use_db_scale"]:
            # 現在の実装ではモデルの出力も正規化されたdBスケールと仮定
            # dBスケールに戻し、さらに線形スケールに戻す
            denormalized_db = utils.denormalize_db_data(predicted_output_normalized, config.PREPROCESS_CONFIG["min_db"], config.PREPROCESS_CONFIG["max_db"])
            # ここではdBスケールのまま保存。必要なら線形に戻す処理を追加。
            final_output = denormalized_db
        else:
            final_output = predicted_output_normalized
            
        # MATLABで扱えるように、(チャープ数, サンプル数)に再転置
        final_output_for_matlab = final_output.T
        
        # 結果を.matファイルとして保存
        output_filename = os.path.basename(test_file).replace("input", "suppressed") + ".mat"
        output_filepath = os.path.join(config.INFERENCE_CONFIG["output_mat_path"], output_filename)
        utils.save_as_mat(output_filepath, {"suppressed_signal": final_output_for_matlab})

if __name__ == '__main__':
    main()