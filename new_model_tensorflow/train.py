# 📁 train.py
# モデルの学習を実行するメインスクリプト

import tensorflow as tf
from tensorflow import keras
import os

# 各モジュールをインポート
import config
import data_loader
import model

def main():
    # 1. データセットの準備
    train_dataset, valid_dataset = data_loader.get_datasets(
        batch_size=config.TRAIN_CONFIG["batch_size"],
        validation_split=config.TRAIN_CONFIG["validation_split"]
    )
    
    # 2. モデルの構築
    interference_suppression_model = model.build_model()
    
    # 3. モデルのコンパイル
    interference_suppression_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.TRAIN_CONFIG["learning_rate"]),
        loss=keras.losses.MeanSquaredError(), # 回帰タスクなので平均二乗誤差を使用
        metrics=['mae'] # 評価指標として平均絶対誤差も監視
    )
    
    # 4. コールバックの準備
    # モデルの保存先ディレクトリを作成
    os.makedirs(config.TRAIN_CONFIG["model_save_path"], exist_ok=True)
    
    callbacks = [
        # 検証データの損失が最も良かったモデルを自動で保存
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(config.TRAIN_CONFIG["model_save_path"], "best_model.keras"),
            save_best_only=True,
            monitor="val_loss"
        ),
        # 学習の進捗をTensorBoardで可視化
        keras.callbacks.TensorBoard(log_dir=config.TRAIN_CONFIG["model_save_path"]),
        # 性能が改善しなくなったら学習を早期終了
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, verbose=1)
    ]
    
    # 5. 学習の実行
    print("\n学習を開始します...")
    history = interference_suppression_model.fit(
        train_dataset,
        epochs=config.TRAIN_CONFIG["epochs"],
        validation_data=valid_dataset,
        callbacks=callbacks
    )
    print("学習が完了しました。")
    print(f"最適なモデルは {config.TRAIN_CONFIG['model_save_path']} に保存されています。")

if __name__ == '__main__':
    main()