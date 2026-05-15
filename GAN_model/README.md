# GAN_model — 干渉あり信号生成 (データ拡張用)

クリーン信号を条件として受け取り、合成干渉信号を生成する条件付き GAN。  
学習データが少ない場合に、干渉あり信号をデータ拡張で増やすことを目的とする。

---

## アーキテクチャ

| モジュール | 内容 |
|---|---|
| **Generator** | U-Net 1D (クリーン信号 + ランダムノイズ → 合成干渉信号) |
| **Critic** | PatchGAN 1D (干渉あり信号が本物か偽物かを識別) |
| **学習方式** | WGAN-GP (Wasserstein 距離 + Gradient Penalty) |

```
クリーン信号 (B, 512, 2)
     │
  Encoder × 4  [stride=2 Conv1d]
     │   512 → 256 → 128 → 64 → 32
     │
  Bottleneck (B, 512, 32)
  + ランダムノイズ z を加算 ← ここで多様な干渉パターンを生成
     │
  Decoder × 4  [ConvTranspose1d + skip connection]
     │   32 → 64 → 128 → 256 → 512
     │
  合成干渉信号 (B, 512, 2)
```

---

## ディレクトリ構成

```
GAN_model/
├── config.py           # データパス・モデル・学習設定
├── model.py            # Generator, Critic, Gradient Penalty
├── data_loader.py      # Dataset / DataLoader
├── utils.py            # テキスト読み込み・正規化関数
├── train.py            # 学習スクリプト (WGAN-GP)
├── generate.py         # データ拡張スクリプト
├── requirements.txt    # 必要パッケージ
│
├── learning_data/      # 学習データ (以下の構造を用意)
│   ├── input/
│   │   ├── real/  real_input_0001.txt ...  ← 干渉あり信号 (実部)
│   │   └── imag/  imag_input_0001.txt ...  ← 干渉あり信号 (虚部)
│   └── label/
│       ├── real/  real_label_0001.txt ...  ← クリーン信号 (実部)
│       └── imag/  imag_label_0001.txt ...  ← クリーン信号 (虚部)
│
├── saved_models/       # 学習済みモデルの保存先 (自動生成)
│   └── GAN_YYYYMMDD_HHMM/
│       ├── G_final.pth
│       ├── D_final.pth
│       └── checkpoint_epoch0020.pth ...
│
└── generated_data/     # 生成データの出力先 (自動生成)
    └── YYYYMMDD_HHMM/
        ├── variation_01/input/real/real_input_0001.txt ...
        ├── variation_01/input/imag/imag_input_0001.txt ...
        ├── variation_02/ ...
        └── ...
```

---

## セットアップ

```bash
pip install -r requirements.txt
```

---

## 使い方

### 1. データの配置

`learning_data/` 以下に `new_model_pytorch` と同じ形式でデータを配置する。  
1 ファイル = 16 チャープ (行) × 512 サンプル (列) のテキスト形式。

### 2. 設定の変更 (config.py)

必要に応じて以下の項目を変更する。

```python
# データパス
DATA_CONFIG = {
    "learning_data_path": ...,  # 学習データのルートディレクトリ
}

# 学習設定
TRAIN_CONFIG = {
    "epochs":       200,    # 学習エポック数
    "batch_size":   64,
    "lr_g":         1e-4,   # Generator 学習率
    "lr_d":         4e-4,   # Critic 学習率
    "n_critic":     5,      # Critic 更新回数 / Generator 1 回
    "lambda_gp":    10.0,   # Gradient Penalty 重み
    "lambda_l1":    0.0,    # L1 Loss 重み (0 = 多様性重視, >0 = 実データに近づける)
    "use_wandb":    True,   # wandb でログを記録するか
    "save_interval": 20,    # 何エポックごとにチェックポイントを保存するか
}

# モデル設定
MODEL_CONFIG = {
    "latent_dim":    64,   # ノイズベクトルの次元数 (大きいほど多様な信号を生成)
    "base_channels": 64,   # Conv1D のベースチャンネル数
}
```

### 3. 学習

```bash
python train.py
```

学習済みモデルは `saved_models/GAN_YYYYMMDD_HHMM/` に保存される。

**学習の目安:**  
- Loss D が安定して小さな値に収束し、Loss G がゆっくり下がれば正常。  
- WGAN-GP では D Loss が負値になることがある (正常動作)。  
- Val L1 が下がらない場合はエポック数を増やすか `lambda_l1` を少し上げる。

### 4. データ拡張 (干渉あり信号の生成)

`config.py` の `GENERATE_CONFIG["trained_model_path"]` に学習済みモデルのパスを設定してから実行する。

```python
# config.py
GENERATE_CONFIG = {
    "trained_model_path": "saved_models/GAN_YYYYMMDD_HHMM",  # ← 実際のパスに変更
    "source_data_path":   "learning_data",                    # クリーン信号の取得元
    "num_variations":     5,   # 1 クリーン信号あたり何種類の干渉信号を生成するか
}
```

```bash
python generate.py
```

`generated_data/YYYYMMDD_HHMM/variation_XX/` 以下に、元データと同じ txt 形式で出力される。

---

## 各ファイルの説明

### config.py
全設定を一元管理するファイル。データパス、モデル構造、学習率などをここで変更する。

### model.py
- `Generator`: U-Net 1D。クリーン信号とノイズから干渉信号を生成する。
- `Critic`: PatchGAN 1D。干渉信号が本物かどうかをパッチ単位で判定する。
- `compute_gradient_penalty`: WGAN-GP の勾配ペナルティ計算。

### data_loader.py
- `label/` (クリーン) を Generator の条件入力として使用。
- `input/` (干渉あり) を Critic の real sample として使用。
- 干渉あり信号の最大絶対値で両方を正規化し、スケール関係を保持する。

### train.py
WGAN-GP の学習ループ。1 エポックごとに以下を実行する。
1. Critic を `n_critic` 回更新 (Wasserstein 距離 + Gradient Penalty)
2. Generator を 1 回更新
3. 検証データで L1 距離を計測してログ出力

### generate.py
学習済み Generator を使い、クリーン信号から複数バリエーションの合成干渉信号を生成する。  
異なるランダムノイズ `z` を使うことで、1 つのクリーン信号から多様な干渉パターンを生成できる。

---

## 注意事項

- **生成時の振幅スケール**: 学習時は干渉信号の max_abs で正規化しているが、生成時はクリーン信号の max_abs を代用するため、振幅が実データとわずかに異なる場合がある。
- **`lambda_l1` の設定**: `0.0` にすると多様性が最大になる。実データに近い信号が欲しい場合は `1.0〜10.0` に設定する。
- **wandb が不要な場合**: `config.py` の `use_wandb` を `False` に設定する。
