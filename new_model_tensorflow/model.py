# 📁 model.py
# Keras API を用いたモデルの定義

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 設定ファイルをインポート
import config

class AttentionBiLSTM(keras.Model):
    def __init__(self, hidden_size, num_heads, dropout_rate, **kwargs):
        super().__init__(**kwargs)
        # 双方向LSTM層
        self.bilstm1 = layers.Bidirectional(layers.LSTM(hidden_size, return_sequences=True))
        
        # Attention層
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=hidden_size)
        self.add = layers.Add()
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        
        # さらに重ねる双方向LSTM層
        self.bilstm2 = layers.Bidirectional(layers.LSTM(hidden_size, return_sequences=True))
        self.bilstm3 = layers.Bidirectional(layers.LSTM(hidden_size, return_sequences=True))
        
        # 出力層（TimeDistributedで各タイムステップにDense層を適用）
        self.dense_out = layers.TimeDistributed(layers.Dense(config.PREPROCESS_CONFIG["num_chirps"]))

    def call(self, inputs, training=False):
        # inputsの形状: (batch, samples, chirps)
        
        # 1層目のBiLSTM
        x = self.bilstm1(inputs)
        
        # Self-Attention + Residual Connection & LayerNorm
        attention_output = self.attention(query=x, value=x, key=x, training=training)
        x = self.layernorm(self.add([x, attention_output]))
        
        # 2, 3層目のBiLSTM
        x = self.bilstm2(x)
        x = self.bilstm3(x)
        
        # 出力層
        output = self.dense_out(x)
        
        return output

def build_model():
    """設定に基づいてモデルを構築して返す"""
    model = AttentionBiLSTM(
        hidden_size=config.MODEL_CONFIG["hidden_size"],
        num_heads=config.MODEL_CONFIG["num_attention_heads"],
        dropout_rate=config.MODEL_CONFIG["dropout_rate"]
    )
    return model