import torch
import torch.nn as nn
import torch.nn.functional as F

class ComplexConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation=None):
        super(ComplexConv2D, self).__init__()
        self.real_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.imag_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = activation

    def forward(self, x):
        real, imag = torch.chunk(x, 2, dim=1)
        real_out = self.real_conv(real) - self.imag_conv(imag)
        imag_out = self.imag_conv(real) + self.real_conv(imag)
        out = torch.cat((real_out, imag_out), dim=1)
        if self.activation:
            out = self.activation(out)
        return out

class ComplexBatchNormalization(nn.Module):
    def __init__(self, num_features):
        super(ComplexBatchNormalization, self).__init__()
        self.real_bn = nn.BatchNorm2d(num_features)
        self.imag_bn = nn.BatchNorm2d(num_features)

    def forward(self, x):
        real, imag = torch.chunk(x, 2, dim=1)
        real = self.real_bn(real)
        imag = self.imag_bn(imag)
        return torch.cat((real, imag), dim=1)

class cReLU(nn.Module):
    def __init__(self):
        super(cReLU, self).__init__()

    def forward(self, x):
        real, imag = torch.chunk(x, 2, dim=1)
        real = F.relu(real)
        imag = F.relu(imag)
        return torch.cat((real, imag), dim=1)

class CV_FCN(nn.Module):
    def __init__(self, depth, filters, kernel_size, use_bn=False, mod=None, sub_connect=False):
        super(CV_FCN, self).__init__()

        self.layers = nn.Sequential(
            ComplexConv2D(1, filters, kernel_size, padding='same'),
            cReLU(),
            *[nn.Sequential(
                ComplexConv2D(filters, filters, kernel_size, padding='same'),
                cReLU(),
            ) for _ in range(depth-2)],
            ComplexConv2D(filters, 1, kernel_size, padding='same'),
        )
    
    def forward(self, x):
        return self.layers(x)

if __name__ == "__main__":
    # Example usage
    depth = 11
    filters = 16
    kernel_size = 3
    use_bn = True
    mod = 'add'
    sub_connect = True

    model = CV_FCN(depth, filters, kernel_size, use_bn=use_bn, mod=mod, sub_connect=sub_connect)
    # input_tensor = torch.randn(16, 11, 256, 256, dtype=torch.float)  # Complex64 in PyTorch
    # output = model(input_tensor)
    # print(output.shape)
    from torchinfo import summary
    summary(model, input_size=(1, 2, 256, 256))

    import time
    batch_size = 1
    size = 256
    warmup_runs = 10
    measure_runs = 100
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # ダミー入力データの作成
    dummy_input = torch.randn(batch_size, 2, size, size, device=device)  # 2チャンネルの入力

    print(f"\nバッチサイズ {batch_size} でのレイテンシー測定を開始します...")

    # ウォームアップ実行
    model.to(device)
    model.eval()
    print(f"{warmup_runs}回のウォームアップを実行中...")
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)

    # 測定実行
    timings = []
    print(f"{measure_runs}回の推論を測定中...")
    with torch.no_grad():
        for i in range(measure_runs):
            if str(device) == 'cuda:0':
                torch.cuda.synchronize() # GPU処理の完了を待つ
            start_time = time.perf_counter()
            
            _ = model(dummy_input)
            
            if str(device) == 'cuda:0':
                torch.cuda.synchronize() # GPU処理の完了を待つ
            end_time = time.perf_counter()
            
            timings.append(end_time - start_time)

    # 結果の計算と表示
    avg_latency_ms = (sum(timings) / measure_runs) * 1000
    std_latency_ms = (torch.tensor(timings).std().item()) * 1000

    print("\n--- 測定結果 ---")
    print(f"平均レイテンシー: {avg_latency_ms:.3f} ms")
    print(f"レイテンシーの標準偏差: {std_latency_ms:.3f} ms")
    print("----------------")