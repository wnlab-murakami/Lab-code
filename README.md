# レーダー干渉抑圧AIモデル

FMCWレーダーの観測信号に含まれる干渉波を深層学習モデルで抑圧するプロジェクトです。

## プロジェクト構成

```
.
├── new_model_pytorch/      # PyTorch版 Attention BiLSTM モデル (推奨)
├── model_tensorflow/       # TensorFlow版 各種モデル
├── new_model_tensorflow/   # TensorFlow版 (新)
├── CNN_model/              # CNNモデル
└── ConvMixer/              # ConvMixerモデル
```

## new_model_pytorch (メインモデル)

Attention付きBiLSTMを用いたレーダー干渉抑圧モデルです。

### ファイル構成

| ファイル | 役割 |
|---|---|
| `config.py` | すべての設定を管理 |
| `model.py` | AIモデルの構造定義 |
| `data_loader.py` | データの前処理と供給 |
| `train.py` | モデルの学習を実行 |
| `inference.py` | 学習済みモデルで推論（干渉抑圧）を実行 |
| `utils.py` | 共通の補助関数 |

### 動作環境

- Python 3.10 推奨
- PyTorch (GPU/CPU)
- numpy, scipy, natsort, tqdm, matplotlib, wandb

### セットアップ

```bash
# 仮想環境の作成と有効化
conda create --name pytorch_env python=3.10
conda activate pytorch_env

# PyTorchのインストール (公式サイトに従って環境に合ったコマンドを実行)
# CPU版の例:
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# その他ライブラリのインストール
pip install -r requirements.txt

# wandbのログイン (初回のみ)
wandb login
```

### 使い方

**1. データ準備**

`config.py` の `DATA_CONFIG` で指定されたパスに学習データ (`.txt` 形式) を配置します。

**2. 学習**

```bash
python train.py
```

学習済みモデルは `saved_models_pytorch/` に `.pth` ファイルとして保存されます。

**3. 推論**

```bash
python inference.py
```

結果は `inference_results_pytorch/` に `.mat` ファイルとして保存されます。

### トラブルシューティング

- `FileNotFoundError`: `config.py` のデータ・モデルパスを確認してください。
- `ModuleNotFoundError`: 仮想環境が有効か、`pip install -r requirements.txt` が完了しているか確認してください。
- CUDAエラー: PyTorch公式サイトでドライバと互換性のあるCUDAバージョンを選択してください。
