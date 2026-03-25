import tensorflow.compat.v1 as tf

tf.compat.v1.disable_v2_behavior()
tf.compat.v1.disable_eager_execution() 

from tensorflow import keras
import preprocess_final_copy
# import output
import random
import os
import pickle
import transformer as encode_model
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time
import copy
import csv
import subprocess as sp
import sys
#import notify 
import tqdm
import wandb
from pathlib import Path

tf.app.flags.DEFINE_string('f', '', 'kernel')
dt_now = datetime.datetime.today()
from tensorflow.keras.utils import plot_model

# パラメータの付与
flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('hidden_size', 200, 'number of hidden cell size')  # 隠れ層のサイズ
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')  # 学習率
flags.DEFINE_integer('batch_size', 128, 'batch size')
flags.DEFINE_integer('all_data_size', 6400, 'all data size')
flags.DEFINE_integer('train_size', 5120, 'train size')
flags.DEFINE_integer('valid_size', 1280, 'valid_size')
flags.DEFINE_integer('num_epoch', 400, 'number of epoch')
flags.DEFINE_boolean('use_clipping', False, 'use clipping')
flags.DEFINE_boolean('use_decay', False, 'exponential decay')  # 指数関数的減衰
flags.DEFINE_boolean('restore', True, 'use restore')  # モデルの呼び出し
flags.DEFINE_boolean('use_fftloss', False, 'use fft loss')
flags.DEFINE_float('fft_coefficient', 0.1, 'learning rate')  # フーリエ係数
flags.DEFINE_string('save_path', '', 'save path')
flags.DEFINE_string('mod', 'train', 'train, null mod')  # 学習の実行
flags.DEFINE_integer('num_layer', 3, 'number of transformer layer')
flags.DEFINE_integer('num_head', 1, 'number of transformer head')
flags.DEFINE_integer('num_dff', 4, 'number of transformer dff')
flags.DEFINE_integer('d_model', 1, 'number of transformer d_model')
flags.DEFINE_integer('drop_rate', 0, 'Drop out rate')
flags.DEFINE_float('rnn_keep_prob', 1, 'rnn drop out rate')
flags.DEFINE_boolean('make_data', True, 'Determine make data')
flags.DEFINE_boolean('median_filter', False, 'Determine make data')
flags.DEFINE_integer('model', 5, 'model_number')
flags.DEFINE_boolean('distributed', False, 'use distributed learning')
flags.DEFINE_boolean('grid_search', False, 'use grid search')

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

# random.seed(2024) 

class Radar:
    def __init__(self, mod=FLAGS.mod, make_data=FLAGS.make_data):  # コンスタンス
        print(FLAGS.use_fftloss)
        print(type(FLAGS.use_fftloss))
        self.save_path = FLAGS.save_path

        if make_data:
            # IF interference is high, use median filter
            self.datas = preprocess_final_copy.data(
                use_median_filter=FLAGS.median_filter)  # preprocess_final_copyにあるdataのインスタンスを生成
            self.max_length = self.datas.max_length
            self.data_length = len(self.datas.inputs)
            self.train_inputs = self.datas.inputs[:self.data_length - FLAGS.valid_size]
            self.train_labels = self.datas.labels[:self.data_length - FLAGS.valid_size]
            self.valid_inputs = self.datas.inputs[self.data_length - FLAGS.valid_size:]
            self.valid_labels = self.datas.labels[self.data_length - FLAGS.valid_size:]
            print('all data size is', self.data_length)
            print('train size is', self.train_inputs.shape)
            print('valid size is', self.valid_inputs.shape)

        # self.max_length = 1024
        self.signal_input = tf.compat.v1.placeholder(tf.float32, [None, self.max_length])
        self.signal_label = tf.compat.v1.placeholder(tf.float32, [None, self.max_length])
        # self.fft_signal_label = tf.placeholder(tf.float32, [None, self.max_length])
        self.rnn_keep_prob = tf.compat.v1.placeholder(tf.float32)
        self.dense_drop_rate = tf.compat.v1.placeholder(tf.float32)
        self.training = tf.compat.v1.placeholder(tf.bool)
        self.attention_training = False
        if mod=='train':
            self.attention_training = True

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        if FLAGS.model == 0:
            model = self.SelfAttentionModel()
            self.save_path = 'model/Attention_biGRU/ckpt'
            if not(os.path.isdir(self.save_path[:-4])):
                os.mkdir(self.save_path[:-4])  # モデルの保存先を指定
            self.restore_path = 'model/Attention_biGRU_max_norm/ckpt'  # 読み込むモデルのパスを指定
        if FLAGS.model == 1:
            model = self.SelfAttentionModel2()
            self.save_path = 'model/GRU_2/ckpt'  # モデルの保存先を指定
            if not(os.path.isdir(self.save_path[:-4])):
                os.mkdir(self.save_path[:-4])
            self.restore_path = 'model/GRU/ckpt'  # 読み込むモデルのパスを指定
        if FLAGS.model == 2:
            model = self.SimpleRNN()
            self.save_path = 'model/RNN_L4_hid200/res'  # モデルの保存先を指定
            if not(os.path.isdir(self.save_path[:-4])):
                os.mkdir(self.save_path[:-4])
            self.restore_path = 'model/RNN_L4_hid200_2/res'  # 読み込むモデルのパスを指定
        if FLAGS.model == 3:
            model = self.LSTM()
            self.save_path = 'model/biLSTM/biLSTM_noise_20250526/ckpt'# モデルの保存先を指定
            if not(os.path.isdir(self.save_path[:-4])):
                os.mkdir(self.save_path[:-4])
            self.restore_path = 'model/biLSTM/biLSTM_noise_20250526/ckpt'  # 読み込むモデルのパスを指定
        if FLAGS.model == 4:
            model = self.Attention_RNN()
            self.save_path = 'model/Attention_RNN/ckpt'# モデルの保存先を指定
            if not(os.path.isdir(self.save_path[:-4])):
                os.mkdir(self.save_path[:-4])
            self.restore_path = 'model/Attention_biRNN/ckpt'  # 読み込むモデルのパスを指定
        if FLAGS.model == 5:
            model = self.Attention_LSTM()
            self.save_path = 'model/Attention_biLSTM/Attention_biLSTM_20250701/res'# モデルの保存先を指定
            target_dir = Path(self.save_path[:-4])
            if not(target_dir.exists()):
                target_dir.mkdir(parents=True, exist_ok=True)
            self.restore_path = 'model/Attention_biLSTM/Attention_biLSTM_20250701/res'  # 読み込むモデルのパスを指定
    
        
        self.optimizer(model)
        
        self.saver = tf.train.Saver()
        if mod == 'train':           
            self.train()
        # if mod == 'train' and grid_search:
        #     self.grid_search()

    def SelfAttentionModel(self):
        new_input = tf.expand_dims(self.signal_input, axis=2)

        with tf.variable_scope('GOU1'):
            cellfw = tf.nn.rnn_cell.GRUCell(FLAGS.hidden_size // 2)
            cellbw = tf.nn.rnn_cell.GRUCell(FLAGS.hidden_size // 2)
            output, _ = tf.nn.bidirectional_dynamic_rnn(cellfw, cellbw, new_input, dtype=tf.float32)

            # 一方向
            # cellfw = tf.nn.rnn_cell.GRUCell(FLAGS.hidden_size)
            # output, _ = tf.nn.dynamic_rnn(cellfw, new_input, dtype=tf.float32)
        rnn_output = tf.concat(output, 2)
        # rnn_output = tf.add(rnn_output, new_input)

        sample_encoder = encode_model.Encoder(num_layers=FLAGS.num_layer, d_model=FLAGS.d_model,
                                            num_heads=FLAGS.num_head,
                                            dff=FLAGS.num_dff)
        sample_encoder_output = sample_encoder(rnn_output, training=self.attention_training, mask=None)

        concat_output = tf.concat([rnn_output, sample_encoder_output], axis=2)
        concat_output = self.layernorm1(concat_output)

        with tf.variable_scope('GOU2'):
            cellfw = tf.nn.rnn_cell.GRUCell(FLAGS.hidden_size // 2)
            cellbw = tf.nn.rnn_cell.GRUCell(FLAGS.hidden_size // 2)
            output, _ = tf.nn.bidirectional_dynamic_rnn(cellfw, cellbw, concat_output, dtype=tf.float32)

            # 一方向
            # cellfw = tf.nn.rnn_cell.GRUCell(FLAGS.hidden_size)
            # output, _ = tf.nn.dynamic_rnn(cellfw, concat_output, dtype=tf.float32)
        rnn_output2 = tf.concat(output, 2)

        # rnn_output2 = tf.add(rnn_output2, rnn_output)

        sample_encoder2 = encode_model.Encoder(num_layers=FLAGS.num_layer, d_model=FLAGS.d_model,
                                            num_heads=FLAGS.num_head,
                                            dff=FLAGS.num_dff)
        sample_encoder_output2 = sample_encoder2(rnn_output2, training=self.attention_training, mask=None)

        concat_output2 = tf.concat([rnn_output2, sample_encoder_output2], axis=2)
        concat_output2 = self.layernorm1(concat_output2)

        with tf.variable_scope('GOU3'):
            cellfw = tf.nn.rnn_cell.GRUCell(FLAGS.hidden_size // 2)
            cellbw = tf.nn.rnn_cell.GRUCell(FLAGS.hidden_size // 2)
            output, _ = tf.nn.bidirectional_dynamic_rnn(cellfw, cellbw, concat_output2, dtype=tf.float32)

            # 一方向
            # cellfw = tf.nn.rnn_cell.GRUCell(FLAGS.hidden_size)
            # output, _ = tf.nn.dynamic_rnn(cellfw, concat_output2, dtype=tf.float32)
        rnn_output3 = tf.concat(output, 2)

        # rnn_output3 = tf.add(rnn_output3, rnn_output2)

        rnn_output3 = tf.expand_dims(rnn_output3, axis=3)

        # conv_output = tf.layers.conv2d(rnn_output3, 1, [1, rnn_output3.shape[2]], strides=[1, 1], padding='valid')
        # squeeze_layer3 = tf.squeeze(conv_output)
        average_layer3 = tf.layers.average_pooling2d(rnn_output3, [1, rnn_output3.shape[2]], strides=[1, 1])
        squeeze_layer3 = tf.squeeze(average_layer3)
        fft_signal = tf.math.l2_normalize(tf.abs(tf.signal.rfft(squeeze_layer3)), axis=1)
        return squeeze_layer3, fft_signal

    def SelfAttentionModel2(self):
        new_input = tf.expand_dims(self.signal_input, axis=2)

        with tf.variable_scope('GOU1'):
            # cellfw = tf.nn.rnn_cell.GRUCell(FLAGS.hidden_size // 2)
            # cellbw = tf.nn.rnn_cell.GRUCell(FLAGS.hidden_size // 2)
            # output, _ = tf.nn.bidirectional_dynamic_rnn(cellfw, cellbw, new_input, dtype=tf.float32)

            # 一方向
            cellfw = tf.nn.rnn_cell.GRUCell(FLAGS.hidden_size)
            output, _ = tf.nn.dynamic_rnn(cellfw, new_input, dtype=tf.float32)
        rnn_output = tf.concat(output, 2)

        with tf.variable_scope('GOU2'):
            # cellfw = tf.nn.rnn_cell.GRUCell(FLAGS.hidden_size // 2)
            # cellbw = tf.nn.rnn_cell.GRUCell(FLAGS.hidden_size // 2)
            # output, _ = tf.nn.bidirectional_dynamic_rnn(cellfw, cellbw, rnn_output, dtype=tf.float32)

            # 一方向
            cellfw = tf.nn.rnn_cell.GRUCell(FLAGS.hidden_size)
            output, _ = tf.nn.dynamic_rnn(cellfw, rnn_output, dtype=tf.float32)
        rnn_output2 = tf.concat(output, 2)

        with tf.variable_scope('GOU3'):
            # cellfw = tf.nn.rnn_cell.GRUCell(FLAGS.hidden_size // 2)
            # cellbw = tf.nn.rnn_cell.GRUCell(FLAGS.hidden_size // 2)
            # output, _ = tf.nn.bidirectional_dynamic_rnn(cellfw, cellbw, rnn_output2, dtype=tf.float32)

            # 一方向
            cellfw = tf.nn.rnn_cell.GRUCell(FLAGS.hidden_size)
            output, _ = tf.nn.dynamic_rnn(cellfw, rnn_output2, dtype=tf.float32)
        rnn_output3 = tf.concat(output, 2)

        rnn_output3 = tf.expand_dims(rnn_output3, axis=3)
        average_layer3 = tf.layers.average_pooling2d(rnn_output3, [1, rnn_output3.shape[2]], strides=[1, 1])
        squeeze_layer3 = tf.squeeze(average_layer3)
        fft_signal = tf.math.l2_normalize(tf.abs(tf.signal.rfft(squeeze_layer3)), axis=1)
        return squeeze_layer3, fft_signal

    def SimpleRNN(self):
        new_input = tf.expand_dims(self.signal_input, axis=2)

        with tf.variable_scope('GOU1'):
            # biRNN
            # cellfw = tf.nn.rnn_cell.BasicRNNCell(FLAGS.hidden_size // 2)
            # cellbw = tf.nn.rnn_cell.BasicRNNCell(FLAGS.hidden_size // 2)
            # output, _ = tf.nn.bidirectional_dynamic_rnn(cellfw, cellbw, new_input, dtype=tf.float32)

            # RNN

            cell = tf.nn.rnn_cell.BasicRNNCell(FLAGS.hidden_size)
            output, _ = tf.nn.dynamic_rnn(cell, new_input, dtype=tf.float32)
        rnn_output = tf.concat(output, 2)

        with tf.variable_scope('GOU2'):
            # biRNN
            # cellfw = tf.nn.rnn_cell.BasicRNNCell(FLAGS.hidden_size // 2)
            # cellbw = tf.nn.rnn_cell.BasicRNNCell(FLAGS.hidden_size // 2)
            # output, _ = tf.nn.bidirectional_dynamic_rnn(cellfw, cellbw, rnn_output, dtype=tf.float32)

            # RNN
            cell = tf.nn.rnn_cell.BasicRNNCell(FLAGS.hidden_size)
            output, _ = tf.nn.dynamic_rnn(cell, rnn_output, dtype=tf.float32)
        rnn_output2 = tf.concat(output, 2)

        with tf.variable_scope('GOU3'):
            # biRNN
            # cellfw = tf.nn.rnn_cell.BasicRNNCell(FLAGS.hidden_size // 2)
            # cellbw = tf.nn.rnn_cell.BasicRNNCell(FLAGS.hidden_size // 2)
            # output, _ = tf.nn.bidirectional_dynamic_rnn(cellfw, cellbw, rnn_output2, dtype=tf.float32)

            # RNN
            cell = tf.nn.rnn_cell.BasicRNNCell(FLAGS.hidden_size)
            output, _ = tf.nn.dynamic_rnn(cell, rnn_output2, dtype=tf.float32)
        rnn_output3 = tf.concat(output, 2)
        
        with tf.variable_scope('GOU4'):
            # biRNN
            # cellfw = tf.nn.rnn_cell.BasicRNNCell(FLAGS.hidden_size // 2)
            # cellbw = tf.nn.rnn_cell.BasicRNNCell(FLAGS.hidden_size // 2)
            # output, _ = tf.nn.bidirectional_dynamic_rnn(cellfw, cellbw, rnn_output3, dtype=tf.float32)

            # RNN
            cell = tf.nn.rnn_cell.BasicRNNCell(FLAGS.hidden_size)
            output, _ = tf.nn.dynamic_rnn(cell, rnn_output3, dtype=tf.float32)
        rnn_output4 = tf.concat(output, 2)

        rnn_output4 = tf.expand_dims(rnn_output4, axis=3)
        average_layer3 = tf.layers.average_pooling2d(rnn_output4, [1, rnn_output4.shape[2]], strides=[1, 1])
        squeeze_layer3 = tf.squeeze(average_layer3)
        fft_signal = tf.math.l2_normalize(tf.abs(tf.signal.rfft(squeeze_layer3)), axis=1)
        return squeeze_layer3, fft_signal

    def LSTM(self):
        new_input = tf.expand_dims(self.signal_input, axis=2)

        with tf.variable_scope('GOU1'):
            # biLSTM
            cellfw = tf.nn.rnn_cell.LSTMCell(FLAGS.hidden_size // 2)
            cellbw = tf.nn.rnn_cell.LSTMCell(FLAGS.hidden_size // 2)
            output, _ = tf.nn.bidirectional_dynamic_rnn(cellfw, cellbw, new_input, dtype=tf.float32)

            # LSTM

            # cell = tf.nn.rnn_cell.LSTMCell(FLAGS.hidden_size)
            # output, _ = tf.nn.dynamic_rnn(cell, new_input, dtype=tf.float32)
        rnn_output = tf.concat(output, 2)

        with tf.variable_scope('GOU2'):
            # biLSTM
            cellfw = tf.nn.rnn_cell.LSTMCell(FLAGS.hidden_size // 2)
            cellbw = tf.nn.rnn_cell.LSTMCell(FLAGS.hidden_size // 2)
            output, _ = tf.nn.bidirectional_dynamic_rnn(cellfw, cellbw, rnn_output, dtype=tf.float32)

            # LSTM
            # cell = tf.nn.rnn_cell.LSTMCell(FLAGS.hidden_size)
            # output, _ = tf.nn.dynamic_rnn(cell, rnn_output, dtype=tf.float32)
        rnn_output2 = tf.concat(output, 2)

        with tf.variable_scope('GOU3'):
            # biLSTM
            cellfw = tf.nn.rnn_cell.LSTMCell(FLAGS.hidden_size // 2)
            cellbw = tf.nn.rnn_cell.LSTMCell(FLAGS.hidden_size // 2)
            output, _ = tf.nn.bidirectional_dynamic_rnn(cellfw, cellbw, rnn_output2, dtype=tf.float32)

            # RNN
            # cell = tf.nn.rnn_cell.LSTMCell(FLAGS.hidden_size)
            # output, _ = tf.nn.dynamic_rnn(cell, rnn_output2, dtype=tf.float32)
        rnn_output3 = tf.concat(output, 2)
        
        with tf.variable_scope('GOU4'):
            # biLSTM
            cellfw = tf.nn.rnn_cell.LSTMCell(FLAGS.hidden_size // 2)
            cellbw = tf.nn.rnn_cell.LSTMCell(FLAGS.hidden_size // 2)
            output, _ = tf.nn.bidirectional_dynamic_rnn(cellfw, cellbw, rnn_output3, dtype=tf.float32)

            # RNN
            # cell = tf.nn.rnn_cell.LSTMCell(FLAGS.hidden_size)
            # output, _ = tf.nn.dynamic_rnn(cell, rnn_output2, dtype=tf.float32)
        rnn_output4 = tf.concat(output, 2)

        rnn_output4 = tf.expand_dims(rnn_output4, axis=3)
        average_layer3 = tf.layers.average_pooling2d(rnn_output4, [1, rnn_output4.shape[2]], strides=[1, 1])
        squeeze_layer3 = tf.squeeze(average_layer3)
        fft_signal = tf.math.l2_normalize(tf.abs(tf.signal.rfft(squeeze_layer3)), axis=1)
        return squeeze_layer3, fft_signal

    def Attention_RNN(self):
        new_input = tf.expand_dims(self.signal_input, axis=2)

        with tf.variable_scope('GOU1'):
            # biRNN
            cellfw = tf.nn.rnn_cell.BasicRNNCell(FLAGS.hidden_size // 2)
            cellbw = tf.nn.rnn_cell.BasicRNNCell(FLAGS.hidden_size // 2)
            output, _ = tf.nn.bidirectional_dynamic_rnn(cellfw, cellbw, new_input, dtype=tf.float32)

            # RNN

            # cell = tf.nn.rnn_cell.BasicRNNCell(FLAGS.hidden_size)
            # output, _ = tf.nn.dynamic_rnn(cell, new_input, dtype=tf.float32)
        rnn_output = tf.concat(output, 2)
        
        sample_encoder = encode_model.Encoder(num_layers=FLAGS.num_layer, d_model=1,
                                            num_heads=FLAGS.num_head,
                                            dff=FLAGS.num_dff)
        sample_encoder_output = sample_encoder(rnn_output, training=self.attention_training, mask=None)
        
        concat_output = tf.concat([rnn_output, sample_encoder_output], axis=2)
        concat_output = self.layernorm1(concat_output)

        with tf.variable_scope('GOU2'):
            # biRNN
            cellfw = tf.nn.rnn_cell.BasicRNNCell(FLAGS.hidden_size // 2)
            cellbw = tf.nn.rnn_cell.BasicRNNCell(FLAGS.hidden_size // 2)
            output, _ = tf.nn.bidirectional_dynamic_rnn(cellfw, cellbw, concat_output, dtype=tf.float32)

            # RNN
            # cell = tf.nn.rnn_cell.BasicRNNCell(FLAGS.hidden_size)
            # output, _ = tf.nn.dynamic_rnn(cell, concat_output, dtype=tf.float32)
        rnn_output2 = tf.concat(output, 2)
        
        sample_encoder2 = encode_model.Encoder(num_layers=FLAGS.num_layer, d_model=1,
                                            num_heads=FLAGS.num_head,
                                            dff=FLAGS.num_dff)
        sample_encoder_output2 = sample_encoder2(rnn_output2, training=self.attention_training, mask=None)

        concat_output2 = tf.concat([rnn_output2, sample_encoder_output2], axis=2)
        concat_output2 = self.layernorm1(concat_output2)

        with tf.variable_scope('GOU3'):
            # biRNN
            cellfw = tf.nn.rnn_cell.BasicRNNCell(FLAGS.hidden_size // 2)
            cellbw = tf.nn.rnn_cell.BasicRNNCell(FLAGS.hidden_size // 2)
            output, _ = tf.nn.bidirectional_dynamic_rnn(cellfw, cellbw, concat_output2, dtype=tf.float32)

            # RNN
            # cell = tf.nn.rnn_cell.BasicRNNCell(FLAGS.hidden_size)
            # output, _ = tf.nn.dynamic_rnn(cell, concat_output2, dtype=tf.float32)
        rnn_output3 = tf.concat(output, 2)
        
        # with tf.variable_scope('GOU4'):
        #     # biRNN
        #     # cellfw = tf.nn.rnn_cell.BasicRNNCell(FLAGS.hidden_size // 2)
        #     # cellbw = tf.nn.rnn_cell.BasicRNNCell(FLAGS.hidden_size // 2)
        #     # output, _ = tf.nn.bidirectional_dynamic_rnn(cellfw, cellbw, rnn_output3, dtype=tf.float32)

        #     # RNN
        #     cell = tf.nn.rnn_cell.BasicRNNCell(FLAGS.hidden_size)
        #     output, _ = tf.nn.dynamic_rnn(cell, rnn_output3, dtype=tf.float32)
        # rnn_output4 = tf.concat(output, 2)

        rnn_output3 = tf.expand_dims(rnn_output3, axis=3)
        average_layer3 = tf.layers.average_pooling2d(rnn_output3, [1, rnn_output3.shape[2]], strides=[1, 1])
        squeeze_layer3 = tf.squeeze(average_layer3)
        fft_signal = tf.math.l2_normalize(tf.abs(tf.signal.rfft(squeeze_layer3)), axis=1)
        return squeeze_layer3, fft_signal

    def Attention_LSTM(self):
        new_input = tf.expand_dims(self.signal_input, axis=2)

        with tf.variable_scope('GOU1'):
            # biLSTM
            cellfw = tf.nn.rnn_cell.LSTMCell(FLAGS.hidden_size // 2)
            cellbw = tf.nn.rnn_cell.LSTMCell(FLAGS.hidden_size // 2)
            output, _ = tf.nn.bidirectional_dynamic_rnn(cellfw, cellbw, new_input, dtype=tf.float32)

            # LSTM

            # cell = tf.nn.rnn_cell.LSTMCell(FLAGS.hidden_size)
            # output, _ = tf.nn.dynamic_rnn(cell, new_input, dtype=tf.float32)
        rnn_output = tf.concat(output, 2)
        
        sample_encoder = encode_model.Encoder(num_layers=FLAGS.num_layer, d_model=1,
                                            num_heads=FLAGS.num_head,
                                            dff=FLAGS.num_dff)
        sample_encoder_output = sample_encoder(rnn_output, training=self.attention_training, mask=None)
        concat_output = tf.concat([rnn_output, sample_encoder_output], axis=2)
        concat_output = self.layernorm1(concat_output)

        with tf.variable_scope('GOU2'):
            # biLSTM
            cellfw = tf.nn.rnn_cell.LSTMCell(FLAGS.hidden_size // 2)
            cellbw = tf.nn.rnn_cell.LSTMCell(FLAGS.hidden_size // 2)
            output, _ = tf.nn.bidirectional_dynamic_rnn(cellfw, cellbw, concat_output, dtype=tf.float32)

            # LSTM
            # cell = tf.nn.rnn_cell.LSTMCell(FLAGS.hidden_size)
            # output, _ = tf.nn.dynamic_rnn(cell, concat_output, dtype=tf.float32)
        rnn_output2 = tf.concat(output, 2)
        
        sample_encoder2 = encode_model.Encoder(num_layers=FLAGS.num_layer, d_model=1,
                                            num_heads=FLAGS.num_head,
                                            dff=FLAGS.num_dff)
        sample_encoder_output2 = sample_encoder2(rnn_output2, training=self.attention_training, mask=None)

        concat_output2 = tf.concat([rnn_output2, sample_encoder_output2], axis=2)
        concat_output2 = self.layernorm1(concat_output2)
        
        with tf.variable_scope('GOU3'):
            # biLSTM
            cellfw = tf.nn.rnn_cell.LSTMCell(FLAGS.hidden_size // 2)
            cellbw = tf.nn.rnn_cell.LSTMCell(FLAGS.hidden_size // 2)
            output, _ = tf.nn.bidirectional_dynamic_rnn(cellfw, cellbw, concat_output2, dtype=tf.float32)

            # RNN
            # cell = tf.nn.rnn_cell.LSTMCell(FLAGS.hidden_size)
            # output, _ = tf.nn.dynamic_rnn(cell, concat_output2, dtype=tf.float32)
        rnn_output3 = tf.concat(output, 2)

        rnn_output3 = tf.expand_dims(rnn_output3, axis=3)
        average_layer3 = tf.layers.average_pooling2d(rnn_output3, [1, rnn_output3.shape[2]], strides=[1, 1])
        squeeze_layer3 = tf.squeeze(average_layer3)
        fft_signal = tf.math.l2_normalize(tf.abs(tf.signal.rfft(squeeze_layer3)), axis=1)
        return squeeze_layer3, fft_signal
    
    def optimizer(self, model):

        if FLAGS.use_clipping:
            self.logits, self.fft_logits = model  # それぞれの変数をSelfAttentionModel関数の返り値として定義(self.logits=squeeze_layer3,self.fft_logits=fft_signal)
            self.loss = tf.reduce_mean(
                tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.logits, self.signal_label)),
                                        reduction_indices=1)))  # 二乗平均誤差
            self.fft_signal_label = tf.math.l2_normalize(tf.abs(tf.signal.rfft(self.signal_label)),
                                                        axis=1)  # 実数値フーリエ変換を行った後、正規化
            self.fft_loss = tf.reduce_mean(
                tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.fft_logits, self.fft_signal_label)),  # 二乗平均誤差
                                        reduction_indices=1)))
            self.sum_loss = self.loss + FLAGS.fft_coefficient * self.fft_loss

            if FLAGS.use_fftloss:
                self.final_loss = self.sum_loss
            else:
                self.final_loss = self.loss

            if FLAGS.use_decay:
                self.global_step = tf.Variable(0, trainable=False)
                learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, self.global_step,
                                                            2000, 0.96, staircase=True)
                optimize = tf.train.AdamOptimizer(learning_rate)
                gvs = optimize.compute_gradients(self.final_loss)

                def ClipIfNotNone(grad):
                    if grad is None:
                        return grad
                    return tf.clip_by_value(grad, -5, 5)

                capped_gvs = [(ClipIfNotNone(grad), var) for grad, var in gvs]
                self.optimizer = optimize.apply_gradients(capped_gvs, global_step=self.global_step)
            else:
                optimize = tf.train.AdamOptimizer(FLAGS.learning_rate)
                gvs = optimize.compute_gradients(self.final_loss)

                def ClipIfNotNone(grad):
                    if grad is None:
                        return grad
                    return tf.clip_by_value(grad, -5, 5)

                capped_gvs = [(ClipIfNotNone(grad), var) for grad, var in gvs]
                self.optimizer = optimize.apply_gradients(capped_gvs)


        else:
            # self.logits, self.fft_logits, self.rnn_output= model
            self.logits, self.fft_logits = model
            self.loss = tf.reduce_mean(
                tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.logits, self.signal_label)), reduction_indices=1)))
            self.fft_signal_label = tf.math.l2_normalize(tf.abs(tf.signal.rfft(self.signal_label)), axis=1)
            self.fft_loss = tf.reduce_mean(
                tf.sqrt(
                    tf.reduce_sum(tf.square(tf.subtract(self.fft_logits, self.fft_signal_label)),
                                    reduction_indices=1)))
            
            self.bias_loss = tf.reduce_sum(self.logits - self.signal_label)
            self.sum_loss = self.loss + FLAGS.fft_coefficient * self.fft_loss  # 損失＋FFTによる損失＋フーリエ係数


            
            if FLAGS.use_fftloss:  # fftlossを考慮する場合はsum_loss、考慮しない場合はlossを使用
                self.final_loss = self.sum_loss
            else:
                self.final_loss = self.loss

            if FLAGS.use_decay:
                self.global_step = tf.Variable(0, trainable=False)
                learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, self.global_step,
                                                            2000, 0.96, staircase=True)
                self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.final_loss,
                                                                                global_step=self.global_step)
            else:
                self.optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(
                    self.final_loss)  # Adamオプティマイザによりfinal_lossを最小化

    def train(self):
        total_step = self.train_inputs.shape[0] // FLAGS.batch_size + 1
        print(self.train_inputs.shape[0])
        # print('total step is %d' % total_step)
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True  # メモリを必要分確保
        min_validation_loss = 10
        best_epoch = 0
        # # TensorBoard用のログディレクトリを作成
        # current_time = time.strftime("%Y%m%d-%H%M%S")
        # train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        # valid_log_dir = 'logs/gradient_tape/' + current_time + '/valid'
        # train_summary_writer = tf.summary.FileWriter(train_log_dir)
        # valid_summary_writer = tf.summary.FileWriter(valid_log_dir)
        wandb.init(
        # set the wandb project where this run will be logged
            project="Radar-RNN",

            # track hyperparameters and run metadata
            config={
            "learning_rate": FLAGS.learning_rate,
            "architecture": "Attention",
            "Layer": 3,
            "dataset": "senario50",
            "epochs": FLAGS.num_epoch,
            "batch":  FLAGS.batch_size,
        }
        )
        with tf.compat.v1.Session(config=config) as sess:

            sess.run(tf.global_variables_initializer())  # 変数の初期化
            if FLAGS.restore == True:
                self.saver.restore(sess, self.restore_path)
            logs = []
            valid_loss = 0
            fft_valid_loss = 0
                # ログディレクトリの作成
            # log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            # file_writer = tf.summary.create_file_writer(log_dir)
            time_start = time.perf_counter()  # 学習時間計測開始
            for epoch in range(FLAGS.num_epoch):
                print('epoch:', epoch)
                for step in tqdm.tqdm(range(total_step)):
                    # print(step)
                    sample = random.sample(range(self.train_inputs.shape[0]), FLAGS.batch_size)  # 入力から、バッチサイズ分の要素をランダムに取り出す
                    train_batch = self.train_inputs[sample]
                    train_label = self.train_labels[sample]
                    sess.run(self.optimizer,
                            feed_dict={self.signal_input: train_batch, self.signal_label: train_label,  # 学習の実行
                                        self.rnn_keep_prob: FLAGS.rnn_keep_prob, self.dense_drop_rate: FLAGS.drop_rate})

                # if step % 100 == 0:
                # print('-----------------------------------------------------------')
                train_loss, fft_train_loss  = sess.run([self.loss, self.fft_loss],
                                    feed_dict={self.signal_input: train_batch, self.signal_label: train_label,
                                                self.rnn_keep_prob: FLAGS.rnn_keep_prob, self.dense_drop_rate: 0})

                for valid_step in range(self.valid_inputs.shape[0] // FLAGS.batch_size): #バッチサイズで割り切れる時注意
                    print("Hello, I'm Mickey Mouse!")

                    valid_batch = self.valid_inputs[valid_step * FLAGS.batch_size:(valid_step + 1) * FLAGS.batch_size]
                    valid_label = self.valid_labels[valid_step * FLAGS.batch_size:(valid_step + 1) * FLAGS.batch_size]


                    
                    valid_loss_batch, fft_valid_loss_batch = sess.run([self.loss, self.fft_loss],
                                        feed_dict={self.signal_input: valid_batch, self.signal_label: valid_label,
                                                    self.rnn_keep_prob: 1, self.dense_drop_rate: 0})
                    if valid_loss_batch == None:
                        print('valid loss is None')
                        break
                    else:
                        print('valid loss is %f' % valid_loss_batch)
                    valid_loss += valid_loss_batch
                    fft_valid_loss += fft_valid_loss_batch                 
                valid_loss /= (self.valid_inputs.shape[0] // FLAGS.batch_size)
                fft_valid_loss /= (self.valid_inputs.shape[0] // FLAGS.batch_size)


                # num_epoch = step * FLAGS.batch_size // self.train_inputs.shape[0]
                print('------------------------------------------------------------')
                print('current epoch is %d' % (epoch+1))
                print('')
                print('train loss is: %f' % train_loss)
                print('fft train loss is: %f' % fft_train_loss)
                print('sum_train loss is: %f' % (train_loss + fft_train_loss))
                print('')
                print('valid loss is: %f' % valid_loss)
                print('fft valid loss is: %f' % fft_valid_loss)
                print('sum_valid real loss is: %f' % (valid_loss + fft_valid_loss))
                print('minimum valid loss is: {0:0.4f} in epoch {1}'.format(min_validation_loss, best_epoch))

                wandb.log({"train_loss": train_loss, "fft_train_loss": fft_train_loss, "total_train_loss": train_loss+fft_train_loss,
                "valid_loss": valid_loss, "fft_valid_loss": fft_valid_loss, "total_valid_loss": valid_loss+fft_valid_loss})
                if (epoch+1) % 5==0 and valid_loss < min_validation_loss:
                    best_epoch = epoch + 1
                    min_validation_loss = valid_loss
                    self.saver.save(sess, self.save_path)
                    print(self.save_path)

                    print('best model saved!!')
                    # rest time calculation 
                print('------------------------------------------------------------')       


                # Logging
                # log = {'epoch': num_epoch, 'train_loss': train_loss + fft_train_loss,
                #         'valid_loss': valid_loss + fft_valid_loss}
                log = {'epoch': epoch, 'train_loss': train_loss,
                        'valid_loss': valid_loss}
                logs.append(log)
                
                # file_writer = tf.summary.create_file_writer(log_dir)
                # with file_writer.as_default():
                #     for log in logs:
                #         tf.summary.scalar('train_loss', log['train_loss'], step=log['epoch'])
                #         tf.summary.scalar('valid_loss', log['valid_loss'], step=log['epoch'])


            time_end = time.perf_counter()
            time_train = time_end - time_start
            print('training time is: %f' % time_train)

            # Save logs
            df = pd.DataFrame(logs)
            Now = dt_now  # 現在時刻
            df.to_csv('learning_log/%d%d%d%drate_%s_batch_%s_data_%s_real' % (
            dt_now.year, dt_now.month, dt_now.day, dt_now.hour, FLAGS.learning_rate, FLAGS.batch_size,
            FLAGS.all_data_size), index=False)

    
        wandb.finish()
        now = dt_now
        # # f = pd.read_csv('../learning_log/%d%d%d%drate_%s_batch_%s_data_%s_real' % (
        # # dt_now.year, dt_now.month, dt_now.day, dt_now.hour, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.all_data_size))
        # # Set values
        # x = df['epoch'].values
        # y0 = df['train_loss'].values
        # y1 = df['valid_loss'].values

        # # Set background color to white
        # fig = plt.figure()
        # fig.patch.set_facecolor('white')

        # # Plot lines
        # plt.xlabel('epoch')
        # plt.ylabel('loss')
        # plt.plot(x, y0, label='train_loss')
        # plt.plot(x, y1, label='valid_loss')
        # plt.legend()

        # Save Figure
        now = dt_now
        # fig.savefig('../learning_log/%d%d%d%d_rate_%s_batch_%s_data_%s_real.png' % (
        # dt_now.year, dt_now.month, dt_now.day, dt_now.hour, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.all_data_size))
        # Visualize
        # plt.show()
    



if __name__ == '__main__':
    Radar()
