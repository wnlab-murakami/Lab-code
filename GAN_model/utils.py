import numpy as np


def load_data_from_txt(filepath: str) -> np.ndarray:
    """テキストファイルを読み込み NumPy 配列として返す。"""
    with open(filepath, "r") as f:
        lines = f.readlines()
    data = [list(map(float, line.strip().split())) for line in lines if line.strip()]
    return np.array(data, dtype=np.float32)


def max_abs_normalize_complex_channels(
    data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    複素数データ (2 チャンネル: 実部・虚部) をサンプルごとに最大絶対値で正規化する。

    Args:
        data: shape (..., sequence_length, 2)
    Returns:
        normalized: shape (..., sequence_length, 2)
        max_abs   : shape (..., 1, 1)  逆正規化に使用
    """
    real = data[..., 0]
    imag = data[..., 1]
    abs_val = np.sqrt(real ** 2 + imag ** 2)
    max_abs = np.max(abs_val, axis=-1, keepdims=True)   # (..., 1)
    max_abs = np.expand_dims(max_abs, axis=-1)           # (..., 1, 1)
    normalized = data / max_abs
    return normalized, max_abs


def max_abs_denormalize_complex_channels(
    normalized: np.ndarray, max_abs: np.ndarray
) -> np.ndarray:
    """最大絶対値正規化の逆変換。"""
    return normalized * max_abs
