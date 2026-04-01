import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# 1. モデルの定義
class Simple2DCNN(nn.Module):
    """
    2次元データ (1024x40) を入力とし、
    同じサイズのクリーンなデータを出力するシンプルなCNNモデル
    """
    def __init__(self):
        super().__init__()
        # --- Encoder ---
        # 2Dデータから階層的に特徴を抽出する
        self.encoder = nn.Sequential(
            # Input: (N, 1, 1024, 40)
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # -> (N, 16, 512, 20)
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # -> (N, 32, 256, 10)
        )
        
        # --- Decoder ---
        # 抽出した特徴から元のサイズのデータを復元する
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2), # -> (N, 16, 512, 20)
            nn.ReLU(),
            
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=2, stride=2),  # -> (N, 1, 1024, 40)
            # U-NetなどではここにSkip Connectionを追加すると性能が向上する
        )

    def forward(self, x):
        # 入力データの次元を確認・追加 (N, H, W) -> (N, 1, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# 2. ダミーデータの生成
def create_dummy_2d_data(num_samples, height, width):
    """
    学習用の2次元ダミーデータを生成する。
    y_true: クリーンな信号（ベースとなるパターン）
    x_noisy: y_trueに干渉（ノイズ）を加えた信号
    """
    # クリーンな信号（ターゲット）
    grid_x, grid_y = torch.meshgrid(torch.linspace(0, 5, height), torch.linspace(0, 5, width), indexing='ij')
    base_pattern = torch.sin(grid_x) + torch.cos(grid_y)
    y_true = base_pattern.unsqueeze(0).repeat(num_samples, 1, 1)

    # 干渉（ノイズ）を加えた信号（入力）
    noise = torch.randn(num_samples, height, width) * 0.5
    x_noisy = y_true + noise
    
    return x_noisy, y_true

# 3. ハイパーパラメータと設定
# データ関連
NUM_SAMPLES = 200
HEIGHT = 1024
WIDTH = 40

# 学習関連
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 1e-3

# 4. メインの学習処理
if __name__ == '__main__':
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # データ生成とデータローダーの準備
    x_train, y_train = create_dummy_2d_data(NUM_SAMPLES, HEIGHT, WIDTH)
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # モデルのインスタンス化
    model = Simple2DCNN().to(device)

    # ★損失関数: 平均二乗誤差 (Mean Squared Error)
    # 画像や信号の復元タスク（回帰）で標準的に使われる
    loss_function = nn.MSELoss()

    # ★最適化関数: Adam
    # パラメータごとに学習率を適応的に調整する高機能な最適化手法
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 学習ループ
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            # データをデバイスに転送
            # モデルへの入力は (N, 1, H, W) であることを想定
            batch_x = batch_x.unsqueeze(1).to(device)
            batch_y = batch_y.unsqueeze(1).to(device)

            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = loss_function(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.6f}')
            
    print("Training finished.")