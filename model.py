import torch
import torch.nn as nn
import torch.optim as optim


class DualModalUNet(nn.Module):
    def __init__(self):
        super(DualModalUNet, self).__init__()
        # Encoder for thermal images
        self.encoder_thermal = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Encoder for visible images
        self.encoder_visible = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(66, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Decoder / Up-sampling
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 4, kernel_size=1),  # 使用4个输出通道
            nn.ReLU()  # 使用ReLU来保持非负激活
        )

        # Global Average Pooling and a Fully Connected Layer
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(4, 4)  # 输入特征4, 输出特征4

    def forward(self, x_thermal, x_visible):
        x_thermal = self.encoder_thermal(x_thermal)
        x_visible = self.encoder_visible(x_visible)

        x = torch.cat((x_thermal, x_visible), dim=1)

        x = self.bottleneck(x)
        x = self.decoder(x)

        # Global Average Pooling
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)  # Flatten the batch
        x = self.fc(x)

        return x

