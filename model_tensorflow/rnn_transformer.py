# --- ライブラリのインポート ---
# TensorFlow 2.x 環境で 1.x系のコードを動かすための互換設定
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution() 

from tensorflow import keras
import preprocess_final_copy
import random
import os
import pickle
import transformer as encode_model # 別のファイルで定義されたTransformerエンコーダをインポート
import numpy as np
import pandas as pd
import tqdm
import wandb # 実験管理ツール Weights & Biases

# --- パラメータ設定 ---
# tf.app.flagsを使って、コマンドラインからも変更可能なグローバル設定を定義
flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('hidden_size', 200, 'RNNの隠れ層のサイズ')
flags.DEFINE_float('learning_rate', 1e-3, '学習率')
flags.DEFINE_integer('batch_size', 128, 'ミニバッチのサイズ')
# ... (その他の多数のフラグ定義) ...
flags.DEFINE_integer('model', 5, '使用するモデルの番号 (0-5)')

# 使用するGPUを環境変数で指定
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# --- メインクラス ---
class Radar:
    def __init__(self, mod=FLAGS.mod, make_data=FLAGS.make_data):
        """
        クラスの初期化メソッド。
        データの読み込み、モデルの構築、オプティマイザの定義などを行う。
        """
        # --- データの準備 ---
        if make_data:
            # preprocess_final_copy.pyのdataクラスを呼び出し、全データをメモリにロード
            self.datas = preprocess_final_copy.data(use_median_filter=FLAGS.median_filter)
            self.max_length = self.datas.max_length
            
            # データを学習用と検証用に分割
            self.train_inputs = self.datas.inputs[:len(self.datas.inputs) - FLAGS.valid_size]
            self.train_labels = self.datas.labels[:len(self.datas.inputs) - FLAGS.valid_size]
            self.valid_inputs = self.datas.inputs[len(self.datas.inputs) - FLAGS.valid_size:]
            self.valid_labels = self.datas.labels[len(self.datas.inputs) - FLAGS.valid_size:]

        # --- TensorFlow計算グラフの定義 ---
        # placeholder: モデルにデータを供給するための「入れ物」を定義
        self.signal_input = tf.compat.v1.placeholder(tf.float32, [None, self.max_length])
        self.signal_label = tf.compat.v1.placeholder(tf.float32, [None, self.max_length])
        # ... (他のplaceholder) ...
        
        # --- モデルの選択と構築 ---
        # FLAGS.modelの値に応じて、使用するモデルの構造を切り替える
        if FLAGS.model == 5:
            model_outputs = self.Attention_LSTM() # 今回はAttention_LSTMモデルが選択される
            # モデルの保存パスと、読み込み元のパスを設定
            self.save_path = 'model/Attention_biLSTM/Attention_biLSTM_20250620/res'
            self.restore_path = 'model/Attention_biLSTM/Attention_biLSTM_20250620/res'
    
        # --- 損失関数と最適化手法の定義 ---
        self.optimizer(model_outputs)
        
        # --- モデルの保存・復元用オブジェクト ---
        self.saver = tf.train.Saver()
        
        # もし `mod` が 'train' なら学習プロセスを開始
        if mod == 'train':           
            self.train()

    # --- モデル構造の定義メソッド ---
    def Attention_LSTM(self):
        """
        Attention付き双方向LSTMモデルの計算グラフを定義する。
        """
        # 入力データを(batch, seq_len, 1)の3次元形状に変換
        new_input = tf.expand_dims(self.signal_input, axis=2)

        # 3層の双方向LSTMを定義。各層はtf.variable_scopeで名前空間を分ける
        with tf.variable_scope('GOU1'):
            cellfw = tf.nn.rnn_cell.LSTMCell(FLAGS.hidden_size // 2) # 順方向セル
            cellbw = tf.nn.rnn_cell.LSTMCell(FLAGS.hidden_size // 2) # 逆方向セル
            output, _ = tf.nn.bidirectional_dynamic_rnn(cellfw, cellbw, new_input, dtype=tf.float32)
        rnn_output = tf.concat(output, 2) # 順方向と逆方向の出力を結合
        
        # TransformerのEncoderブロックをAttentionとして使用
        sample_encoder = encode_model.Encoder(...)
        sample_encoder_output = sample_encoder(rnn_output, ...)
        # 残差接続 (Residual Connection) と層正規化 (Layer Normalization)
        concat_output = tf.concat([rnn_output, sample_encoder_output], axis=2)
        concat_output = self.layernorm1(concat_output)

        # 2層目、3層目も同様の構造
        with tf.variable_scope('GOU2'):
            # ... (2層目のBiLSTM + Attention) ...
        with tf.variable_scope('GOU3'):
            # ... (3層目のBiLSTM) ...
        
        # --- 出力層 ---
        # 最終出力を(batch, seq_len, 1)の形状に変換
        rnn_output3 = tf.expand_dims(rnn_output3, axis=3)
        # 最後の隠れ状態の次元を平均化して、出力を(batch, seq_len)の形状にする
        average_layer3 = tf.layers.average_pooling2d(rnn_output3, [1, rnn_output3.shape[2]], strides=[1, 1])
        squeeze_layer3 = tf.squeeze(average_layer3)
        
        # FFT損失計算用の信号も出力
        fft_signal = tf.math.l2_normalize(tf.abs(tf.signal.rfft(squeeze_layer3)), axis=1)
        return squeeze_layer3, fft_signal
    
    def optimizer(self, model):
        """
        損失関数とオプティマイザ（学習方法）の計算グラフを定義する。
        """
        # モデルの出力を受け取る
        self.logits, self.fft_logits = model
        
        # --- 損失の定義 ---
        # 1. 時間領域での損失 (平均二乗誤差)
        self.loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.logits, self.signal_label)), reduction_indices=1)))
        # 2. 周波数領域での損失 (FFT後のスペクトルに対する平均二乗誤差)
        self.fft_signal_label = tf.math.l2_normalize(tf.abs(tf.signal.rfft(self.signal_label)), axis=1)
        self.fft_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.fft_logits, self.fft_signal_label)), reduction_indices=1)))
        
        # 3. 最終的な損失 (FFT損失を加えるかどうかをフラグで切り替え)
        if FLAGS.use_fftloss:
            self.final_loss = self.loss + FLAGS.fft_coefficient * self.fft_loss
        else:
            self.final_loss = self.loss

        # --- 学習率の減衰設定 (オプション) ---
        if FLAGS.use_decay:
            # ...

        # --- 勾配クリッピング (勾配爆発を防ぐための工夫) ---
        if FLAGS.use_clipping:
            optimize = tf.train.AdamOptimizer(FLAGS.learning_rate)
            # 勾配を計算
            gvs = optimize.compute_gradients(self.final_loss)
            # 勾配が-5から5の範囲に収まるようにクリッピング
            capped_gvs = [(tf.clip_by_value(grad, -5, 5), var) for grad, var in gvs]
            # クリッピング後の勾配で重みを更新
            self.optimizer = optimize.apply_gradients(capped_gvs)
        else:
            # クリッピングなしの場合
            self.optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(self.final_loss)

    def train(self):
        """
        実際の学習ループを実行する。
        """
        # ... (設定、変数初期化) ...
        wandb.init(...) # wandbの初期化
        
        with tf.compat.v1.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer()) # 変数を初期化
            
            # エポック数だけループ
            for epoch in range(FLAGS.num_epoch):
                # ステップ数だけループ (1エポック内のミニバッチ処理)
                for step in tqdm.tqdm(range(total_step)):
                    # 学習データからランダムにミニバッチを作成
                    sample = random.sample(range(self.train_inputs.shape[0]), FLAGS.batch_size)
                    train_batch = self.train_inputs[sample]
                    train_label = self.train_labels[sample]
                    
                    # オプティマイザを実行し、モデルの重みを1ステップ更新
                    sess.run(self.optimizer, feed_dict={...})

                # --- 検証フェーズ ---
                # 検証データで現在のモデルの性能を評価
                valid_loss = 0
                for valid_step in range(...):
                    # ...
                    valid_loss_batch, _ = sess.run([self.loss, self.fft_loss], feed_dict={...})
                    valid_loss += valid_loss_batch
                
                # --- 結果の表示と保存 ---
                print(...) # 訓練ロスと検証ロスを表示
                
                # wandbにメトリクスを記録
                wandb.log({"train_loss": train_loss, "valid_loss": valid_loss})
                
                # もし検証ロスが過去最良なら、モデルを保存
                if valid_loss < min_validation_loss:
                    self.saver.save(sess, self.save_path)

            # ... (学習時間、ログの保存) ...
        wandb.finish() # wandbの実行を終了