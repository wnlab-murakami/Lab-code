# 📁 preprocess_final_copy.py
# (コメント追加版)

import glob
import numpy as np
import random
import pickle 
from natsort import natsorted

# --- ファイル読み込み関数 ---
def make_data(filepath):
    """
    単一のテキストファイルを読み込み、数値の2次元リストとして返す。
    各行が1つのチャープのサンプルデータに対応する。
    """
    with open(filepath) as f:
        lines = f.readlines()
    data = []
    for si in lines:
        # 行をスペースで分割し、各要素を浮動小数点数に変換してリストに追加
        data.append([float(i) for i in si.strip().split(' ')])
    return data

# --- データファイルのパスを取得する関数群 ---
def make_input():
    """学習用の入力データ（干渉あり）のファイルパスリストを取得する"""
    files = natsorted(glob.glob('py/learning_data/2025_6_20/input/input_1_*.txt*'))
    # print(f'input files: {len(files)}')
    return files

def make_label():
    """学習用の正解ラベルデータ（干渉なし）のファイルパスリストを取得する"""
    files = natsorted(glob.glob('py/learning_data/2025_6_20/label/label_1_*.txt'))
    return files

def make_inputs_and_labels(input_path, label_path):
    """
    入力とラベルのファイルパスリストを受け取り、全てのデータを読み込んで
    一つの大きなリストにまとめる。
    """
    inputs = []
    labels = []
    for i in range(len(input_path)):
        input_data = make_data(input_path[i])
        label_data = make_data(label_path[i])
        # extendを使って、リストの要素をそのまま追加していく
        inputs.extend(input_data)
        labels.extend(label_data)
    return inputs, labels

# --- データセット全体を管理するクラス ---
class data():
    def __init__(self, use_median_filter=True, train=True):
        """
        クラスがインスタンス化された際に、データの読み込みから前処理までを自動で実行する。
        Args:
            use_median_filter (bool): メディアンフィルタを適用するかどうか
            train (bool): データをシャッフルするかどうか（学習時か評価時か）
        """
        self.use_median_filter = use_median_filter
        
        # 1. 全てのファイルパスを取得
        self.input_path = make_input()
        self.label_path = make_label()
        
        # 2. 全てのファイルを読み込み、巨大なリストに格納
        self.inputs, self.labels = make_inputs_and_labels(self.input_path, self.label_path)
        
        # 3. PythonのリストをNumPy配列に変換
        self.inputs = np.array(self.inputs)
        self.labels = np.array(self.labels)
        print(self.inputs.shape)

        # 4. データセット内の最長サンプル長を記録 (現在は未使用)
        self.max_length = self.test_max_length(self.inputs)

        # 5. 正規化処理を適用
        self.inputs, self.labels = self.normalize_array(self.inputs, self.labels)
        
        if use_median_filter:
            print('Use median filter')
        else:
            print('Not use median filter')
            
        # 6. 学習時であれば、データの順序をランダムにシャッフル
        #    これにより、モデルがデータの順序を学習してしまうのを防ぐ
        if train:
            # 0からデータ数-1までのインデックスリストを作成
            x = list(range(len(self.inputs)))
            random.shuffle(x) # インデックスをシャッフル
            # シャッフルされたインデックスを使って、入力とラベルを同じ順序で並べ替える
            self.inputs = self.inputs[x]
            self.labels = self.labels[x]
            print('This is test', self.inputs.shape)
    
        print('finished loading data!!')
        
    def test_max_length(self, data):
        """データセット内の最長のサンプル長を計算する (現在は未使用)"""
        maxlength = 0
        for i in range(len(data)):
            if len(data[i]) > maxlength:
                maxlength = len(data[i])
        return maxlength    
    
    def median_filter(self, inputs):
        """
        メディアンフィルタを適用して外れ値（極端に大きな値）を除去する。
        信号の中央値の100倍を超える値を0にすることで、大きな干渉スパイクなどを抑制する。
        """
        for idx, signal in enumerate(inputs):
            thres = 100 * np.median(np.abs(signal))
            signal[np.abs(signal) > thres] = 0
            inputs[idx] = signal
        return inputs
    
    def normalize_array(self, inputs, labels):
        """
        入力とラベルデータを正規化する。
        各信号の最大絶対値でその信号の全要素を割ることで、値を-1から1の範囲に収める。
        """
        norm_input = []
        norm_label = []
        if self.use_median_filter:
            inputs = self.median_filter(inputs)
        
        for idx in range(len(inputs)):
            # 各信号の最大絶対値を計算 (L2ノルムではない)
            # 信号内の値の2乗の最大値の平方根 = 最大絶対値
            norm_val = np.sqrt(max(inputs[idx]**2))

            # 計算した最大絶対値で信号全体を割り、正規化
            norm_input.append(inputs[idx] / norm_val)
            # labelもinputと同じ値で正規化する
            norm_label.append(labels[idx] / norm_val)
        
        return np.array(norm_input), np.array(norm_label)