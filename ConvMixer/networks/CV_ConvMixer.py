import torch
import torch.nn as nn
from complexNN.nn import cConv2d, cBatchNorm2d, cGelu

class Residual(nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)

class CV_ConvMixer(nn.Module):
    """Complex-Valued ConvMixer model."""
    def __init__(self, channel, dim, depth, kernel_size=9, patch_size=7):
        super(CV_ConvMixer, self).__init__()
        
        self.layers = nn.Sequential(
            cConv2d(channel, dim, kernel_size=patch_size, padding="same",bias=False),
            cGelu(),
            cBatchNorm2d(dim),
            * [nn.Sequential(
                Residual(nn.Sequential(
                    cConv2d(dim, dim, kernel_size=kernel_size, groups=dim, padding="same",bias=False),
                    cGelu(),
                    cBatchNorm2d(dim)
                )),
                cConv2d(dim, dim, kernel_size=1, bias=False),
                cGelu(),
                cBatchNorm2d(dim)
            ) for _ in range(depth)],
            
            cConv2d(dim, channel, kernel_size=3, padding="same")
        )
        self._initialize_weights() # 重みの初期化を呼び出す

    def _initialize_weights(self):
        """Initializes weights of the model."""
        for m in self.modules():
            if isinstance(m, cConv2d):
                # He initialization for convolutional layers
                if hasattr(m, 'real_conv'):
                    nn.init.kaiming_normal_(m.real_conv.weight, mode='fan_out', nonlinearity='relu')
                    if m.real_conv.bias is not None:
                        nn.init.constant_(m.real_conv.bias, 0)
                if hasattr(m, 'imag_conv'):
                    nn.init.kaiming_normal_(m.imag_conv.weight, mode='fan_out', nonlinearity='relu')
                    if m.imag_conv.bias is not None:
                        nn.init.constant_(m.imag_conv.bias, 0)
            elif isinstance(m, cBatchNorm2d):
                # Initialize BatchNorm with weight 1 and bias 0
                if hasattr(m, 'real_bn'):
                    nn.init.constant_(m.real_bn.weight, 1)
                    nn.init.constant_(m.real_bn.bias, 0)
                if hasattr(m, 'imag_bn'):
                    nn.init.constant_(m.imag_bn.weight, 1)
                    nn.init.constant_(m.imag_bn.bias, 0)

    def forward(self, x):
        return self.layers(x)
    

if __name__ == "__main__":
    # Example usage:
    # Let's assume input has 2 channels (1 real, 1 imaginary).
    # The `channel` parameter in CV_ConvMixer should represent the number of *complex* channels.
    # So if you have 1 complex input channel, the input tensor will have 2 actual channels (real + imag).
    model = CV_ConvMixer(channel=1, dim=16, depth=5)

    # Summary of the model
    from torchinfo import summary
    summary(model, input_size=(1, 1, 256, 256))
    # Example of latency measurement
    import time
    batch_size = 1
    height, width = 257, 256
    warmup_runs = 10
    measure_runs = 100
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Dummy input data creation
    dummy_input = torch.randn(batch_size, 1, height, width, device=device, dtype=torch.cfloat)  # 2-channel input

    print(f"\nStarting latency measurement with batch size {batch_size}...")

    # Warm-up runs
    model.to(device)
    model.eval()
    print(f"Running {warmup_runs} warm-up iterations...")
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