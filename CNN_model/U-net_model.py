import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 1. モデルの定義 (U-Net)
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        self.enc1 = self.conv_block(in_channels, 16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc2 = self.conv_block(16, 32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = self.conv_block(32, 64)
        self.upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(64, 32)
        self.upconv1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(32, 16)
        self.out_conv = nn.Conv2d(16, out_channels, kernel_size=1)

    def conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.upconv2(b)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        return self.out_conv(d1)

# 2. ダミーデータの生成
def create_dummy_2d_data(num_samples, height, width):
    grid_x, grid_y = torch.meshgrid(torch.linspace(0, 5, height), torch.linspace(0, 5, width), indexing='ij')
    base_pattern = torch.sin(grid_x * 2) * torch.cos(grid_y * 1.5)
    y_true = base_pattern.unsqueeze(0).repeat(num_samples, 1, 1)
    noise = torch.randn(num_samples, height, width) * 0.5
    x_noisy = y_true + noise
    return x_noisy, y_true

# 3. ハイパーパラメータと設定
NUM_SAMPLES, HEIGHT, WIDTH = 200, 1024, 40
BATCH_SIZE, EPOCHS, LEARNING_RATE = 16, 30, 1e-3

# 4. メインの学習処理
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    x_train, y_train = create_dummy_2d_data(NUM_SAMPLES, HEIGHT, WIDTH)
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = UNet().to(device)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # ... (学習ループは変更なし) ...
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_x, batch_y in train_loader:
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

    # ======================================================
    # 5. 学習結果の可視化 (ここからが追加部分)
    # ======================================================
    print("Visualizing results...")
    model.eval() # モデルを評価モードに切り替え
    with torch.no_grad(): # 勾配計算を無効化
        # データローダーから1バッチ分のデータを取得
        sample_x, sample_y = next(iter(train_loader))
        sample_x = sample_x.unsqueeze(1).to(device)
        
        # モデルによる予測を実行
        predicted_y = model(sample_x)
        
        # データをCPUに戻し、Numpy配列に変換
        sample_x_np = sample_x.squeeze().cpu().numpy()
        predicted_y_np = predicted_y.squeeze().cpu().numpy()
        sample_y_np = sample_y.cpu().numpy()
        
        # 表示するサンプル数を指定 (最大5つ)
        num_to_show = min(5, BATCH_SIZE)
        
        # サンプルごとに3つの画像をプロット
        for i in range(num_to_show):
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # 共通のカラーバー範囲を設定
            vmin = min(sample_x_np[i].min(), sample_y_np[i].min())
            vmax = max(sample_x_np[i].max(), sample_y_np[i].max())
            
            # 1. ノイズあり入力
            im1 = axes[0].imshow(sample_x_np[i], cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
            axes[0].set_title('Noisy Input')
            
            # 2. モデルの復元出力
            im2 = axes[1].imshow(predicted_y_np[i], cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
            axes[1].set_title('Model Output')
            
            # 3. クリーンな正解データ
            im3 = axes[2].imshow(sample_y_np[i], cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
            axes[2].set_title('Ground Truth')
            
            # カラーバーを追加
            fig.colorbar(im3, ax=axes.ravel().tolist(), shrink=0.7)
            
            plt.suptitle(f'Sample {i+1}')
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()