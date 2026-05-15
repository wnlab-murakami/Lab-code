"""学習済み Generator を使ってクリーン信号から合成干渉信号を生成するスクリプト。

使い方:
    1. config.py の GENERATE_CONFIG["trained_model_path"] に
       学習済みモデルのディレクトリ (G_final.pth があるパス) を設定する。
    2. python generate.py を実行する。

出力構造:
    generated_data/YYYYMMDD_HHMM/
        variation_01/input/real/real_input_XXXX.txt
        variation_01/input/imag/imag_input_XXXX.txt
        variation_02/...
        ...

各 variation は異なるランダムノイズ z を使って生成されるため、
同じクリーン信号から異なる干渉パターンが得られる。

正規化に関する注意:
    学習時は「干渉あり信号の max_abs」でクリーン信号も正規化しているが、
    生成時には干渉あり信号が存在しないため「クリーン信号自身の max_abs」を使用する。
    そのため生成された干渉信号の振幅は実データとわずかに異なる場合がある。
"""

import glob
import os

import numpy as np
import torch
from natsort import natsorted

import config
import model
import utils


def generate() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"デバイス: {device}")

    # --- 学習済み Generator の読み込み ---
    G, _ = model.build_models()
    model_path = os.path.join(config.GENERATE_CONFIG["trained_model_path"], "G_final.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"学習済みモデルが見つかりません: {model_path}\n"
            "config.py の GENERATE_CONFIG['trained_model_path'] を確認してください。"
        )
    G.load_state_dict(torch.load(model_path, map_location=device))
    G.eval().to(device)
    print(f"モデルを読み込みました: {model_path}")

    # --- クリーン信号ファイルの取得 ---
    base        = config.GENERATE_CONFIG["source_data_path"]
    label_real  = os.path.join(base, config.DATA_CONFIG["label_dir_name"], config.DATA_CONFIG["real_dir_name"])
    label_imag  = os.path.join(base, config.DATA_CONFIG["label_dir_name"], config.DATA_CONFIG["imag_dir_name"])

    real_label_files = natsorted(glob.glob(os.path.join(label_real, "real_label_*.txt")))
    if not real_label_files:
        raise FileNotFoundError(f"クリーン信号が見つかりません: {label_real}")

    imag_label_files = [
        f.replace(label_real, label_imag).replace("real_", "imag_")
        for f in real_label_files
    ]

    latent_dim      = config.MODEL_CONFIG["latent_dim"]
    chirps_per_file = config.PREPROCESS_CONFIG["chirps_per_file"]
    num_variations  = config.GENERATE_CONFIG["num_variations"]
    out_base        = config.GENERATE_CONFIG["output_path"]

    print(
        f"\n{len(real_label_files)} ファイル × {num_variations} variation を生成します..."
    )

    for var_idx in range(num_variations):
        var_tag = f"variation_{var_idx + 1:02d}"
        out_real = os.path.join(out_base, var_tag, config.DATA_CONFIG["input_dir_name"], config.DATA_CONFIG["real_dir_name"])
        out_imag = os.path.join(out_base, var_tag, config.DATA_CONFIG["input_dir_name"], config.DATA_CONFIG["imag_dir_name"])
        os.makedirs(out_real, exist_ok=True)
        os.makedirs(out_imag, exist_ok=True)

        for file_idx, (rf, imf) in enumerate(zip(real_label_files, imag_label_files)):
            # クリーン信号を読み込む: shape (chirps, 512)
            real_lb = utils.load_data_from_txt(rf)
            imag_lb = utils.load_data_from_txt(imf)
            clean = np.stack([real_lb, imag_lb], axis=-1).astype(np.float32)  # (16, 512, 2)

            # クリーン信号自身の max_abs で正規化
            clean_norm, max_abs = utils.max_abs_normalize_complex_channels(clean)  # (16, 512, 2)

            clean_tensor = torch.from_numpy(clean_norm).to(device)   # (16, 512, 2)

            # ランダムノイズで干渉信号を生成
            z = torch.randn(chirps_per_file, latent_dim, device=device)
            with torch.no_grad():
                fake_norm = G(clean_tensor, z)   # (16, 512, 2)

            # 逆正規化して元のスケールに戻す
            fake_np = fake_norm.cpu().numpy()                     # (16, 512, 2)
            fake    = utils.max_abs_denormalize_complex_channels(fake_np, max_abs)

            # テキストファイルとして保存 (1 行 = 1 チャープ, 512 列)
            file_num = file_idx + 1
            np.savetxt(
                os.path.join(out_real, f"real_input_{file_num:04d}.txt"),
                fake[:, :, 0],
                fmt="%.6e",
            )
            np.savetxt(
                os.path.join(out_imag, f"imag_input_{file_num:04d}.txt"),
                fake[:, :, 1],
                fmt="%.6e",
            )

        print(f"  {var_tag} 完了")

    print(f"\n生成完了。出力先: {out_base}")


if __name__ == "__main__":
    generate()
