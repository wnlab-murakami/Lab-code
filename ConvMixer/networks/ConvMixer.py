import sys
import torch
import torch.nn as nn
from torchinfo import summary

class Residual(nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)

class ConvMixerLayer(nn.Module):
    def __init__(self, channel, dim, depth, kernel_size=9, patch_size=7):
        super(ConvMixerLayer, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channel, dim, kernel_size=patch_size, padding="same",bias=True),
            nn.GELU(),
            nn.BatchNorm2d(dim),
            *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same",bias=True),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1,bias=True),
                nn.GELU(),
                nn.BatchNorm2d(dim)
            ) for _ in range(depth)],
            nn.Conv2d(dim, channel, kernel_size=3, padding="same",bias=True)
        )
        self._initialize_weights()

    # He weight initialization
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.layers(x)

if __name__ == "__main__":
    model = ConvMixerLayer(2, 32, 5)

    summary(model, input_size=(1, 2, 256, 256))

    import time
    batch_size = 32
    height, width = 256, 256
    warmup_runs = 10
    measure_runs = 100
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # ダミー入力データの作成
    dummy_input = torch.randn(batch_size, 2, height, width, device=device)  # 2チャンネルの入力

    print(f"\nバッチサイズ {batch_size} でのレイテンシー測定を開始します...")

    # ウォームアップ実行
    model.to(device)
    model.eval()
    print(f"{warmup_runs}回のウォームアップを実行中...")
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)

    # Measurement runs
    print(f"Measuring latency over {measure_runs} iterations...")
    start_time = time.time()
    with torch.no_grad():
        for _ in range(measure_runs):
            _ = model(dummy_input)
    
    elapsed_time = time.time() - start_time
    average_latency = elapsed_time / measure_runs * 1000  # Convert to milliseconds
    print(f"Average latency: {average_latency:.4f} ms")